import argparse
import logging
import os,sys
from typing import Type
import random 
from tqdm import tqdm

import torch
import numpy as np
from torch import nn, optim

from sicgan.config import Config
from sicgan.models import Pixel2MeshHead
from sicgan.models import GraphConvClf
from sicgan.data.build_data_loader import build_data_loader
from sicgan.models import MeshLoss
from sicgan.utils.torch_utils import save_checkpoint
from tensorboardX import SummaryWriter



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
    trn_dataloader = build_data_loader(_C, "MeshVox", split_name='train')
   
    
    print("Training Samples: "+str(len(trn_dataloader)))
    #print("Validation Samples: "+str(len(val_dataloader)))
    
    ## Models
    G = Pixel2MeshHead(_C).cuda()
    D = GraphConvClf(_C).cuda()
    
    # Losses
    loss_fn_kwargs = {
        "chamfer_weight": _C.G.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "normal_weight": _C.G.MESH_HEAD.NORMAL_LOSS_WEIGHT,
        "edge_weight": _C.G.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "gt_num_samples": _C.G.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": _C.G.MESH_HEAD.PRED_NUM_SAMPLES,
    }
    
    mesh_loss = MeshLoss(**loss_fn_kwargs).cuda()
    clf_loss = nn.BCELoss().cuda()
    
    ## Optimizers
    G_optimizer = torch.optim.Adam(G.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), lr= 0.0002, betas=(0.5, 0.999))
    
    ## Tensorboard
    tb = SummaryWriter(os.path.join(_C.CKP.experiment_path,'tensorboard')) 

    args  = {}
    args['EXPERIMENT_NAME'] =  _C.EXPERIMENT_NAME
    args['full_experiment_name'] = _C.CKP.full_experiment_name
    args['experiment_path'] = _C.CKP.experiment_path
    args['best_recon_loss'] = _C.CKP.best_loss
    
    # --------------------------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------------------------
    step = 0
    total_step = len(trn_dataloader)
    print('\n ***************** Training *****************')
    
    for epoch in range(_C.SOLVER.NUM_EPOCHS):
        # --------------------------------------------------------------------------------------------
        #   TRAINING 
        # --------------------------------------------------------------------------------------------
        running_loss = 0.0
        print('Epoch: '+str(epoch))
        
        G.train()
        D.train()
    
        for data in tqdm(trn_dataloader):
            step += 1
            imgs = data[0].cuda()
            meshes = data[1].cuda()
        
            ## Update D network
            with torch.no_grad():
                meshes_G = G(imgs)
            D_optimizer.zero_grad()
            D_neg = D(meshes_G)
            D_pos = D(meshes)
            
            loss_D = 0.5*(clf_loss(D_neg, torch.zeros(D_neg.size()).cuda()) + 
                          clf_loss(D_pos, torch.ones(D_pos.size()).cuda()))
            loss_D.backward(retain_graph=True)
            D_optimizer.step()
            
            ## Update G network
            G_optimizer.zero_grad()
            meshes_G = G(imgs)
            recon_loss, _ = mesh_loss(meshes_G, meshes)
            loss_G = recon_loss + clf_loss(D_neg, torch.ones(D_neg.size()).cuda())
            loss_G.backward()
            G_optimizer.step()
            
        
            tb.add_scalar('data/loss_G', loss_G.item(), step)
            tb.add_scalar('data/loss_D', loss_D.item(), step)
            tb.add_scalar('data/Reconstruction_loss', recon_loss.item(), step)
            if _C.OVERFIT:
                if step%10==0:
                    break

        print("===> Epoch[{}]: Loss_D: {:.4f} Loss_G: {:.4f}".format(epoch, loss_D.item(), loss_G.item()))
        
        # Final save of the model
        args = save_checkpoint(G = G,
                               D = D,
                               curr_epoch = epoch,
                               G_loss = loss_G.item(),
                               D_loss = loss_D.item(),
                               recon_loss = recon_loss.item(),
                               curr_step = step,
                               args = args,
                               filename = ('model@epoch%d.pkl' %(epoch)))
          
        print('---------------------------------------------------------------------------------------\n')
    print('Finished Training')
    tb.close() 