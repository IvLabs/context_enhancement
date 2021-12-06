import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel 
from _config import Config as config
from _utils import Helper

class Model(nn.Module):
    def __init__(self, 
                 projector_layers: str, 
                 mbert_out_size: int,
                 transformer_enc: nn.TransformerEncoder,
                 lambd: float): #eg. projector_layers = "1024-1024-1024"
        super().__init__()
        self.projector_layers = projector_layers
        self.mbert_out_size = mbert_out_size
        self.transformer_enc = transformer_enc
        self.lambd = lambd

        self.mbert = BertModel.from_pretrained(config.TOKENIZER)

        sizes = [self.mbert_out_size] + list(map(int, self.projector_layers.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False) #not sure about this one, will have to check about size and all
    
    def forward(self,
                x: torch.tensor,
                y: torch.tensor): #x = numericalised text


        x = self.mbert(x)
        x = self.transformer_enc(x["last_hidden_state"]) #considering sentence representation as the starting special token
        x = torch.sum(x, dim=1)/x.shape[1]

        y = self.mbert(y)
        y = self.transformer_enc(y["last_hidden_state"]) #considering sentence representation as the starting special token
        y = torch.sum(y, dim=1)/y.shape[1]

        x = self.bn(self.projector(x)) #x = [batch_size, projector]
        y = self.bn(self.projector(y)) #y = [batch_size, projector]

        #emperical cross-correlation mattrix 
        c = x.T @ y

        # for multi-gpu: sum cross correlation matrix between all gpus 
        #(uncomment below 2 lines      
        '''
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)
        
        '''
 
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = Helper.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss 