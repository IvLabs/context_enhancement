import torch 
import torch.nn as nn 
from torch import Tensor
from translation_utils import TokenEmbedding
from torch.nn import Transformer
from barlow_utils import off_diagonal


class Translator(nn.Module): 
        
    def __init__(self,
            mbert, 
            transformer,
            tgt_vocab_size: int, 
            emb_size: int):                 

        super(Translator, self).__init__()
    
        self.transformer = transformer
        self.generator = nn.Linear(emb_size, tgt_vocab_size) 
        self.mbert = mbert 
        self.tok_emb = TokenEmbedding(emb_size = emb_size, mbert=self.mbert)
        # self.trg_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        # self.positional_encoding = PositionalEncoding(emb_size, dropout=args.dropout)


    def forward(self, 
            src: Tensor, 
            tgt: Tensor, 
            src_mask: Tensor,
            tgt_mask: Tensor,  
            src_padding_mask: Tensor,
            tgt_padding_mask: Tensor, 
            memory_key_padding_mask: Tensor): 

        # print(src.shape, tgt.shape)
        src_emb = self.tok_emb(src)
        trg_emb = self.tok_emb(tgt)
        out = self.transformer(src_emb, trg_emb, src_mask, tgt_mask, None, 
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask) 
        
        return self.generator(out) 

    def encode(self, 
            src: Tensor, 
            src_mask: Tensor): 
        return self.transformer.encoder(self.tok_emb(src),  src_mask) 

    def decode(self, tgt: Tensor, 
            memory: Tensor, 
            tgt_mask: Tensor): 
        return self.transformer.decoder(self.tok_emb(tgt), memory, tgt_mask)

class BarlowTwins(nn.Module):
    def __init__(self, 
                 projector_layers: str, 
                 mbert_out_size: int,
                 transformer_enc: nn.TransformerEncoder,
                 mbert, 
                 lambd: float): #eg. projector_layers = "1024-1024-1024"
        super().__init__()
        self.projector_layers = projector_layers 
        self.mbert_out_size = mbert_out_size
        self.transformer_enc = transformer_enc
        self.lambd = lambd

        self.mbert = mbert 

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


        x = x.squeeze(-1)
        #print(x.shape)
        x = self.mbert(x)
        x = self.transformer_enc(x["last_hidden_state"].permute(1,0,2)) 
        x = torch.sum(x, dim=0)/x.shape[1] # using avg pooling 
        
        y = y.squeeze(-1)
        y = self.mbert(y)
        y = self.transformer_enc(y["last_hidden_state"].permute(1,0,2)) 
        y = torch.sum(y, dim=0)/y.shape[1] # using avg pooling 

        x = self.projector(x) #x = [batch_size, projector]
        x = self.bn(x)
        # print(x.shape)
        y = self.projector(y)
        y = self.bn(y) #y = [batch_size, projector]

        batch_size = y.shape[0] 
        #emperical cross-correlation mattrix 
        c = x.T @ y

        # for multi-gpu: sum cross correlation matrix between all gpus 
        #(uncomment below 2 lines      
        
        c.div_(batch_size)
        torch.distributed.all_reduce(c)
 
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return c, loss 

