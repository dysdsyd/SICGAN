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
from sicgan.models import GraphConvClf, MeshEncoder
from sicgan.models import encoder_head
from sicgan.data.build_data_loader import build_data_loader
from sicgan.models import MeshLoss
from sicgan.utils.torch_utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter


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
    val_dataloader = build_data_loader(_C, "MeshVox", split_name='val')
    
    print('Training Samples: '+ str(len(trn_dataloader)))
    print('Validation Samples: '+ str(len(val_dataloader)))
    
    
    ## Models
    E = encoder_head(_C).cuda()
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
    E_optimizer = torch.optim.Adam(E.parameters(), lr= 0.02, betas=(0.5, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr= 0.002, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), lr= 0.02, betas=(0.5, 0.999))
    
    ## Tensorboard
    tb = SummaryWriter(os.path.join('tensorboard/', _C.CKP.full_experiment_name)) 

    # --------------------------------------------------------------------------------------------
    #   TRAINING LOOP
    # --------------------------------------------------------------------------------------------
    step = 0
    total_step = len(trn_dataloader)
    best_loss = 1000
    print('\n ***************** Training *****************')
    
    for epoch in range(_C.SOLVER.NUM_EPOCHS):
        # --------------------------------------------------------------------------------------------
        #   TRAINING 
        # --------------------------------------------------------------------------------------------
        E_losses = []
        D_losses = []
        G_losses = []
        EG_losses = []
        trn_losses = []
        val_losses = []
        print('Epoch: '+str(epoch))
        
        E.train()
        G.train()
        D.train()
    
        for data in tqdm(trn_dataloader):
            step += 1
            imgs = data[0].cuda()
            meshes = data[1].cuda()
        
            
            z, means, sigmas = E(imgs)
            meshes_G = G(imgs,z)
            
            D_optimizer.zero_grad()
            
            ## Update D network
            with torch.no_grad():
                D_neg = D(meshes_G)
                
            D_pos = D(meshes)
            
            loss_D = 0.5*(clf_loss(D_neg, torch.zeros(D_neg.size()).cuda()) + 
                          clf_loss(D_pos, torch.ones(D_pos.size()).cuda()))
            loss_D.backward(retain_graph=True)
            D_optimizer.step()

            #Update E&G Network
            E_optimizer.zero_grad()
            G_optimizer.zero_grad()

            loss_kl_E = torch.mean(0.5 * torch.sum(torch.exp(sigmas) + means**2 - 1. - sigmas, 1))
            recon_loss, _ = mesh_loss(meshes_G, meshes)
            loss_G = recon_loss + clf_loss(D_neg, torch.ones(D_neg.size()).cuda())
            loss_EG = loss_kl_E + loss_G
            loss_EG.backward()
            E_optimizer.step()
            G_optimizer.step()

            
            D_losses.append(loss_D.item())
            G_losses.append(loss_G.item())
            E_losses.append(loss_kl_E.item())
            EG_losses.append(loss_EG.item())
            trn_losses.append(recon_loss.item())
            
            
            if _C.OVERFIT:
                if step%10==0:
                    break

        # ----------------------------------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------------------------------
        D.eval()
        G.eval()
        E.eval()
        print("\n\n\tEvaluating..")
        for i, data in enumerate(tqdm(val_dataloader), 0):
            imgs = data[0].cuda()
            meshes = data[1].cuda()
            
            
            with torch.no_grad():
                z_v,_,_ = E(imgs)
                meshes_G = G(imgs,z_v) # -----Check which z you have to consider here!-------
                val_loss, _ = mesh_loss(meshes_G, meshes)
            val_losses.append(val_loss.item())
            
            
            
            
        print("===> Epoch[{}]: Loss_D: {:.4f} Loss_G: {:.4f} Loss_E: {:.4f} Loss_Recon: {:.4f}".format(epoch, np.mean(D_losses), np.mean(G_losses), np.mean(E_losses), np.mean(trn_losses)))
        tb.add_scalar('data/loss_G', np.mean(G_losses), epoch)
        tb.add_scalar('data/loss_D', np.mean(D_losses), epoch)
        tb.add_scalar('data/loss_E', np.mean(E_losses), epoch)
        tb.add_scalar('data/Training_loss', np.mean(trn_losses), epoch) 
        tb.add_scalar('data/Validation_Loss',  np.mean(val_losses), epoch)
            
        if np.mean(val_losses)<best_loss:
            best_loss = np.mean(val_losses)
            torch.save(D.state_dict(),  os.path.join(_C.CKP.experiment_path,'D.pth'))
            torch.save(G.state_dict(),  os.path.join(_C.CKP.experiment_path,'G.pth'))
            torch.save(E.state_dict(),  os.path.join(_C.CKP.experiment_path,'E.pth'))
          
        print('---------------------------------------------------------------------------------------\n')
    print('Finished Training')
    tb.close() 