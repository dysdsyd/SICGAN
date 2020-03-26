import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.ops import GraphConv
from sicgan.config import Config
from pytorch3d.structures.utils import packed_to_list, list_to_padded

class GraphConvClf(nn.Module):
    def __init__(self, cfg):
        super(GraphConvClf, self).__init__()
        input_dim = cfg.D.INPUT_MESH_FEATS
        hidden_dims = cfg.D.HIDDEN_DIMS 
        classes = cfg.D.CLASSES
        gconv_init = cfg.D.CONV_INIT
        
        # Graph Convolution Network
        self.gconvs = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            self.gconvs.append(GraphConv(dims[i], dims[i+1], init=gconv_init, directed=False))
            
        self.fc1 = nn.Linear(dims[-1], 1024)
        self.fc2 = nn.Linear(1024, classes)
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        
        for gconv in self.gconvs:
            verts = F.relu(gconv(verts, edges))
        
        ### VERTS ###
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = verts_idx.unique(return_counts=True)[1]
        verts_packed = packed_to_list(verts, tuple(verts_size))
        verts_padded = list_to_padded(verts_packed)
        
        out  = torch.sum(verts_padded, 1)/verts_size.view(-1,1)
        out = F.relu(self.fc1(out)) 
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out