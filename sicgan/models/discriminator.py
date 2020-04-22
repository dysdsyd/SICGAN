import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.ops import GraphConv
from pytorch3d.structures.utils import packed_to_list, list_to_padded
from sicgan.models.layers import *
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
            
        self.fc1 = nn.Linear(dims[-1], dims[-1])
        self.fc2 = nn.Linear(dims[-1], dims[-1])
        self.fc3 = nn.Linear(dims[-1], classes)
        
        
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
        
#         out  = torch.sum(verts_padded, 1)/verts_size.view(-1,1)
        out = torch.max(verts_padded, 1, keepdim=True)[0]
        out = F.relu(self.fc1(out)) 
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
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
    
    # normalizes symetric, binary adj matrix such that sum of each row is 1 
    def normalize_adj(self, mx):
        rowsum = mx.sum(1)
        r_inv = (1./rowsum).view(-1)
        r_inv[r_inv != r_inv] = 0.
        mx = torch.mm(torch.eye(r_inv.shape[0]).to(mx.device)*r_inv, mx)
        return mx

    def calc_adj(self, faces): 
        max_v = int(faces.max())
        min_v = int(faces.min())
        v1 = faces[:, 0] - min_v
        v2 = faces[:, 1] - min_v
        v3 = faces[:, 2] - min_v
        num_verts = max_v - min_v
        adj = torch.eye(num_verts+1).to(faces.device)

        adj[(v1, v2)] = 1 
        adj[(v1, v3)] = 1 

        adj[(v2, v1)] = 1
        adj[(v2, v3)] = 1 

        adj[(v3, v1)] = 1
        adj[(v3, v2)] = 1 
        adj = self.normalize_adj(adj.cpu())
        return adj

    def forward(self, mesh, play = False):
        # print positions[:5, :5]
        latent = []
        verts_idx = mesh.verts_packed_to_mesh_idx()
        verts_size = tuple(verts_idx.unique(return_counts=True)[1])
        
        faces_idx = mesh.faces_packed_to_mesh_idx()
        faces_size = tuple(faces_idx.unique(return_counts=True)[1])
        
        verts = packed_to_list(mesh.verts_packed(), verts_size)
        faces = packed_to_list(mesh.faces_packed(), faces_size)
        for positions, face in zip(verts, faces):
            adj = self.calc_adj(face).cuda()
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
            latent.append(self.reduce(features , adj, F.elu).unsqueeze(0))
        
        latent = torch.cat(latent, dim=0)
        return latent