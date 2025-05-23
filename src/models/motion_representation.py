import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import copy
from models.layers.motion_encoder import VQEncoderV3, VQDecoderV3, LocalEncoder
from models.utils.skeleton import build_edge_topology
from models.layers.utils import reparameterize

# ----------- AE, VAE ------------- #

class VAEConv(nn.Module):
    def __init__(self, args):
        super(VAEConv, self).__init__()
        self.encoder = VQEncoderV3(args)
        self.decoder = VQDecoderV3(args)
        self.fc_mu = nn.Linear(args.vae_length, args.vae_length)
        self.fc_logvar = nn.Linear(args.vae_length, args.vae_length)
        self.variational = args.variational
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        mu, logvar = None, None
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        rec_pose = self.decoder(pre_latent)
        return {
            "poses_feat":pre_latent,
            "rec_pose": rec_pose,
            "pose_mu": mu,
            "pose_logvar": logvar,
            }
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        if self.variational:
            mu = self.fc_mu(pre_latent)
            logvar = self.fc_logvar(pre_latent)
            pre_latent = reparameterize(mu, logvar)
        return pre_latent
    
    def decode(self, pre_latent):
        rec_pose = self.decoder(pre_latent)
        return rec_pose

class VAESKConv(VAEConv):
    def __init__(self, args):
        super(VAESKConv, self).__init__(args)
        smpl_fname = args.data_path_1+'smplx_models/smplx/SMPLX_NEUTRAL_2020.npz'
        smpl_data = np.load(smpl_fname, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)
        self.encoder = LocalEncoder(args, edges)
        self.decoder = VQDecoderV3(args)
    