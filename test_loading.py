import numpy as np
from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

import torch
from torch import nn, optim 
from torch.nn import Transformer 
import torchtext
import t_dataset
from t_dataset import  Translation_dataset_t
from t_dataset import  MyCollate
import translation_utils 
from translation_utils import TokenEmbedding, PositionalEncoding 
from translation_utils import create_mask
from transformers import BertModel 
from transformers import AutoTokenizer
from torch import Tensor
from torchtext.data.metrics import bleu_score
from models import Translator
from models import BarlowTwins


#import barlow
#os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
os.environ['WANDB_START_METHOD'] = 'thread'

MANUAL_SEED = 4444

random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description = 'Translation') 

# Training hyper-parameters: 
parser.add_argument('--workers', default=4, type=int, metavar='N', 
                    help='number of data loader workers') 
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=4, type=int, metavar='n',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.2, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--dropout', default=0.01, type=float, metavar='d',
                    help='dropout for training translation transformer')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--clip', default=1, type=float, metavar='GC',
                    help='Gradient Clipping')
parser.add_argument('--betas', default=(0.9, 0.98), type=tuple, metavar='B',
                    help='betas for Adam Optimizer')
parser.add_argument('--eps', default=1e-9, type=float, metavar='E',
                    help='eps for Adam optimizer')
parser.add_argument('--loss_fn', default='cross_entropy', type=str, metavar='LF',
                    help='loss function for translation')

# Transformer parameters: 
parser.add_argument('--dmodel', default=768, type=int, metavar='T', 
                    help='dimension of transformer encoder')
parser.add_argument('--nhead', default=4, type= int, metavar='N', 
                    help= 'number of heads in transformer') 
parser.add_argument('--dfeedforward', default=256, type=int, metavar='F', 
                    help= 'dimension of feedforward layer in transformer encoder') 
parser.add_argument('--nlayers', default=3, type=int, metavar= 'N', 
                   help='number of layers of transformer encoder') 
parser.add_argument('--projector', default='768-256', type=str,
                    metavar='MLP', help='projector MLP')

# Tokenizer: 
parser.add_argument('--tokenizer', default='bert-base-multilingual-uncased', type=str, 
                metavar='T', help= 'tokenizer')
parser.add_argument('--mbert-out-size', default=768, type=int, metavar='MO', 
                    help='Dimension of mbert output')
# Paths: 
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

# to load or barlow or not: 
parser.add_argument('--load', default=0, type=int,
                    metavar='DIR', help='to load barlow twins encoder or not')

# calculate bleu: 
parser.add_argument('--checkbleu', default=5 , type=int,
                    metavar='BL', help='check bleu after these number of epochs')
# train or test dataset
parser.add_argument('--train', default=True , type=bool,
                    metavar='T', help='selecting train set')

parser.add_argument('--print_freq', default=5 , type=int,
                    metavar='PF', help='frequency of printing and saving stats')

''' NOTE: 
        Transformer and tokenizer arguments would remain constant in training and context enhancement step.  
'''

args = parser.parse_args()
# print(args.load)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
dataset = Translation_dataset_t(train=args.train) 
src_vocab_size = dataset.de_vocab_size
trg_vocab_size = dataset.en_vocab_size
tokenizer = dataset.tokenizer  
pad_idx = tokenizer.pad_token_id
sos_idx = tokenizer.cls_token_id 
eos_idx = tokenizer.sep_token_id


mbert = BertModel.from_pretrained('bert-base-multilingual-uncased')
transformer = Transformer(d_model=args.dmodel, 
                              nhead=args.nhead, 
                              num_encoder_layers=args.nlayers, 
                              num_decoder_layers = args.nlayers, 
                              dim_feedforward=args.dfeedforward, 
                              dropout=args.dropout)
model = Translator(mbert=mbert, transformer= transformer, tgt_vocab_size=trg_vocab_size, emb_size=args.mbert_out_size)

####################################################
#################LOAD MODEL#########################
ckpt = torch.load(args.checkpoint_dir/ 'translation_checkpoint.pth', 
                            map_location='cpu')
model.load_state_dict(ckpt['model'])


