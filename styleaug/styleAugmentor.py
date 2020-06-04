#source of base model: https://github.com/philipjackson/style-augmentation paper: https://arxiv.org/abs/1809.05375 "Style Augmentation: Data Augmentation via Style Randomization"

import torch
import torch.nn as nn

from .transformer import Transformer
import numpy as np
import sys
from os.path import join, dirname

#using cuda simultaneously with the classifier could cause occasional crashes, using cpu is more stable but somewhat slower
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class StyleAugmentor(nn.Module):
    def __init__(self):
        super(StyleAugmentor,self).__init__()

        # create transformer:
        self.transformer = Transformer()
        self.transformer.to(device)

        # load checkpoints:
        checkpoint_transformer = torch.load(join(dirname(__file__),'checkpoints/checkpoint_transformer.pth'))
        checkpoint_embeddings = torch.load(join(dirname(__file__),'checkpoints/checkpoint_embeddings.pth'))
        
        # load weights for transformer:
        self.transformer.load_state_dict(checkpoint_transformer['state_dict_ghiasi'],strict=False)

        # load mean imagenet embedding:
        self.imagenet_embedding = checkpoint_embeddings['imagenet_embedding_mean'] # mean style embedding for ImageNet
        self.imagenet_embedding = self.imagenet_embedding.to(device)

        # get mean and covariance of PBN style embeddings:
        self.mean = checkpoint_embeddings['pbn_embedding_mean']
        self.mean = self.mean.to(device) # 1 x 100
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']
        
        # compute SVD of covariance matrix:
        u, s, _ = np.linalg.svd(self.cov.numpy())
        
        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float().to(device) # 100 x 100
    

    def forward(self,x):
        base = self.imagenet_embedding.float()
        
        # sample a random embedding
        embedding = torch.mm(torch.randn(x.size(0),100).to(device),self.A.transpose(1,0)) + self.mean
        
        # interpolate style embeddings:
        embedding = 0.5*embedding + 0.5*base
        
        restyled = self.transformer(x,embedding)
        
        return restyled.detach()
