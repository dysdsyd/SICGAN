import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.ops import GraphConv
from pytorch3d.structures.utils import packed_to_list, list_to_padded
from layers import *
import torchvision.models as models
from pytorch3d.ops import GraphConv



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
    
    

class MeshEncoder(nn.Module):
    def __init__(self, latent_length):
        super(MeshEncoder, self).__init__()
        self.h1 = ZERON_GCN(3, 60)
        self.h21 = ZERON_GCN(60, 60)
        self.h22 = ZERON_GCN(60, 60)
        self.h23 = ZERON_GCN(60, 60)
        self.h24 = ZERON_GCN(60,120)
        self.h3 = ZERON_GCN(120, 120)
        self.h4 = ZERON_GCN(120, 120)
        self.h41 = ZERON_GCN(120, 150)
        self.h5 = ZERON_GCN(150, 200)
        self.h6 = ZERON_GCN(200, 210)
        self.h7 = ZERON_GCN(210, 250)
        self.h8 = ZERON_GCN(250, 300)
        self.h81 = ZERON_GCN(300, 300)
        self.h9 = ZERON_GCN(300, 300)
        self.h10 = ZERON_GCN(300, 300)
        self.h11 = ZERON_GCN(300, 300)
        self.reduce = GCNMax(300,latent_length)
    
    def resnet( self, features, res):
        temp = features[:,:res.shape[1]]
        temp = temp + res
        features = torch.cat((temp,features[:,res.shape[1]:]), dim = 1)
        return features, features
    
    def calc_adj(self, faces): 
        v1 = faces[:, 0]
        v2 = faces[:, 1]
        v3 = faces[:, 2]
        num_verts = int(faces.max())
        adj = torch.eye(num_verts+1).to(faces.device)

        adj[(v1, v2)] = 1 
        adj[(v1, v3)] = 1 

        adj[(v2, v1)] = 1
        adj[(v2, v3)] = 1 

        adj[(v3, v1)] = 1
        adj[(v3, v2)] = 1 

        return adj

    def forward(self, meshes, play = False):
        # print positions[:5, :5]
        latent = []
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        verts = packed_to_list(mesh.verts_packed(), verts_size)
        faces = packed_to_list(mesh.faces_packed(), verts_size)
        for positions, face in zip(verts, faces): 
            adj = self.calc_adj(face)
            features = self.h1(positions, adj, F.elu)
            features = self.h21(features, adj, F.elu)
            features = self.h22(features, adj, F.elu)
            features = self.h23(features, adj, F.elu)
            features = self.h24(features, adj, F.elu)
            features = self.h3(features, adj, F.elu)
            features = self.h4(features, adj, F.elu)
            features = self.h41(features, adj, F.elu)
            features = self.h5(features, adj, F.elu)
            features = self.h6(features, adj, F.elu)
            features = self.h7(features, adj, F.elu)
            features = self.h8(features, adj, F.elu)
            features = self.h81(features, adj, F.elu)
            features = self.h9(features, adj, F.elu)
            features = self.h10(features, adj, F.elu)
            features = self.h11(features, adj, F.elu)
            latent.append(self.reduce(features , adj, F.elu))
        
        latent = torch.cat(latent, dim=0)
        return latent