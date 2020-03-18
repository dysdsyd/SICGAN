import argparse
import logging
import os,sys
from typing import Type
import random 
from tqdm import tqdm

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from style_transfer.data.datasets import ShapenetDataset
from style_transfer.models.base_nn import GraphConvClf
from style_transfer.config import Config
from style_transfer.utils.torch_utils import train_val_split, save_checkpoint, accuracy

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser("Run training for a particular phase.")
parser.add_argument(
    "--config-yml", required=True, help="Path to a config file for specified phase."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the results directory.",
)


logger: logging.Logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    # --------------------------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # --------------------------------------------------------------------------------------------
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml, _A.config_override)

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Create serialization directory and save config in it.
    os.makedirs(_C.CKP.experiment_path, exist_ok=True)
    _C.dump(os.path.join(_C.CKP.experiment_path, "config.yml"))
    
    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    _C.DEVICE = device
    
    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER & CRITERION
    # --------------------------------------------------------------------------------------------
    ## Datasets
    trn_objs, val_objs = train_val_split(config=_C)
    collate_fn = ShapenetDataset.collate_fn
    
    if _C.OVERFIT:
        trn_objs, val_objs = trn_objs[:10], val_objs[:10]
    
    trn_dataset = ShapenetDataset(_C, trn_objs)
    trn_dataloader = DataLoader(trn_dataset, 
                            batch_size=_C.OPTIM.BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    
    val_dataset = ShapenetDataset(_C, val_objs)
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=_C.OPTIM.VAL_BATCH_SIZE, 
                            shuffle=True, 
                            collate_fn=collate_fn, 
                            num_workers=_C.OPTIM.WORKERS)
    
    print("Training Samples: "+str(len(trn_dataloader)))
    print("Validation Samples: "+str(len(val_dataloader)))

    model = GraphConvClf(_C).cuda()
    model.load_state_dict(torch.load('results/exp_03_16_11_22_19_10classes/model@epoch3.pkl')['state_dict'])

#     optimizer = optim.SGD(
#         model.parameters(),
#         lr=_C.OPTIM.LR,
#         momentum=_C.OPTIM.MOMENTUM,
#         weight_decay=_C.OPTIM.WEIGHT_DECAY,
#     )
    optimizer = optim.Adam(
        model.parameters(),
        lr=_C.OPTIM.LR,
    )
#     lr_scheduler = optim.lr_scheduler.LambdaLR(  # type: ignore
#         optimizer, lr_lambda=lambda iteration: 1 - iteration / _C.OPTIM.NUM_ITERATIONS
#     )

    criterion = nn.CrossEntropyLoss()
    args  = {}
    args['EXPERIMENT_NAME'] =  _C.EXPERIMENT_NAME
    args['full_experiment_name'] = _C.CKP.full_experiment_name
    args['experiment_path'] = _C.CKP.experiment_path
    args['best_loss'] = _C.CKP.best_loss
    args['best_acc'] = _C.CKP.best_acc
    # --------------------------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------------------------
    total_step = len(trn_dataloader)
    print('\n ***************** Training *****************')
    for epoch in tqdm(range(4, _C.OPTIM.EPOCH)):
        # --------------------------------------------------------------------------------------------
        #   TRAINING 
        # --------------------------------------------------------------------------------------------
        running_loss = 0.0
        print('Epoch: '+str(epoch))
        model.train()
        
        for i, data in enumerate(tqdm(trn_dataloader), 0):
            if data[0] == None and data[1] == None:
                continue
            label = data[0].cuda()
            mesh = data[1].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(mesh)
            #print(outputs, label)
            if outputs.size()[0] == label.size()[0]:
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                #lr_scheduler.step()
                # print statistics
                running_loss += loss.item()
            else:
                print('Shape Mismatch')
                print(outputs.size(), label.size())
                print(mesh.verts_packed_to_mesh_idx().unique(return_counts=True)[1])
        running_loss /= len(trn_dataloader)
        print('\n\tTraining Loss: '+ str(running_loss))
        
        # ----------------------------------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        print("\n\n\tEvaluating..")
        for i, data in enumerate(tqdm(val_dataloader), 0):
            if data[0] == None and data[1] == None:
                continue
            label = data[0].cuda()
            mesh = data[1].cuda()
            with torch.no_grad():
                batch_prediction = model(mesh)
                if batch_prediction.size()[0] == label.size()[0]:
                    loss = criterion(batch_prediction, label)
                    acc = accuracy(batch_prediction, label)
                    val_loss += loss.item()
                    val_acc += np.sum(acc)
                else:
                    print('Shape Mismatch')
                    print(batch_prediction.size(), label.size())
                    print(mesh.verts_packed_to_mesh_idx().unique(return_counts=True)[1])
        # Average out the loss
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        print('\n\tValidation Loss: '+str(val_loss))
        print('\tValidation Acc: '+str(val_acc.item()))
        # Final save of the model
        args = save_checkpoint(model      = model,
                             optimizer  = optimizer,
                             curr_epoch = epoch,
                             curr_loss  = val_loss,
                             curr_step  = (total_step * epoch),
                             args       = args,
                             curr_acc   = val_acc.item(),
                             trn_loss   = running_loss,
                             filename   = ('model@epoch%d.pkl' %(epoch)))
          
        print('---------------------------------------------------------------------------------------\n')
    print('Finished Training')
    print('Best Accuracy on validation',args['best_acc'])
    print('Best Loss on validation',args['best_loss'])