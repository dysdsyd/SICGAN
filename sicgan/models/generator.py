import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from sicgan.models.backbone import build_backbone
from sicgan.models.heads import MeshRefinementHead
from sicgan.utils.coords import get_blender_intrinsic_matrix

class Pixel2MeshHead(nn.Module):
    def __init__(self, cfg):
        super(Pixel2MeshHead, self).__init__()

        # fmt: off
        backbone                = cfg.G.BACKBONE
        self.ico_sphere_level   = cfg.G.MESH_HEAD.ICO_SPHERE_LEVEL
        # fmt: on

        self.K = get_blender_intrinsic_matrix()
        # backbone
        self.backbone, feat_dims = build_backbone(backbone)
        # mesh head
        cfg.G.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs, z):    # z is the latent vector sampled from P(z|x)
        N = imgs.shape[0]
        device = imgs.device

        img_feats = self.backbone(imgs)
        # concat_feats = torch.cat((img_feats,z),dim=1)
        P = self._get_projection_matrix(N, device)

        init_meshes = ico_sphere(self.ico_sphere_level, device).extend(N)
        refined_meshes = self.mesh_head(img_feats, z, init_meshes, P, subdivide=True)[0]
        return refined_meshes
