import torch
import torch.nn as nn 
import  math 
import transformers 
from transformers import BertModel 


model = BertModel.from_pretrained("bert-base-uncased")

def vocab(en_list: torch.tensor): 
    # a = list(tokenizer(en_list, padding=True, return_tensors='pt')[
    #     'input_ids']) 
    a = torch.stack(en_list)
    a = a.flatten()
    a = a.tolist()
    a = set(a) 
    return a, len(a) 



'''
returns dictionary mapping bert id to an index from 0 to vocab_size
input = set of vocab
return = dict : key (bert_index)  
                value (label from 0 to vocab) 
'''
def bert2id(de_list: set): 
    label_dict = {}
    for n, i in enumerate(de_list): 
        label_dict[i] = n
    
    return label_dict

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_idx):

    PAD_IDX = pad_idx

    # print(src.shape, tgt.shape)
    if len(src.shape) == 1: 
        src = src.unsqueeze(-1)
        tgt = tgt.unsqueeze(-1)
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, emb_size, mbert):
        super(TokenEmbedding, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding = mbert
        # for param in self.embedding.parameters():
        #   param.requires_grad = False
        # for param in self.embedding.pooler.parameters():
        #   param.requires_grad = True
        self.emb_size = emb_size

    def forward(self, tokens: torch.tensor):
        # print(tokens.shape)
        if len(tokens.shape) ==1: 
            tokens  = tokens.unsqueeze(-1)
        return self.embedding(tokens.long().T)['last_hidden_state'].permute(1, 0, 2) * math.sqrt(self.emb_size)
