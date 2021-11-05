#Working around
import torchtext.data as data
#import torchtext.data.Datasets as datasets
import torch 
import torch.nn as nn
from torchtext.datasets import Multi30k 
from transformers import BertModel, BertTokenizer
#from transformers import AutoTokenizer, AutoModelForMaskedLM
from torchtext.data import Field, BucketIterator

bert = BertModel.from_pretrained('bert-base-uncased')

#tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
#xlm_roberta = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")


SRC_FIELD = data.Field(lower=True, batch_first=True, init_token = '<src_sos>', eos_token='<src_eos>', tokenize=bert )
TRG_FIELD = data.Field(lower=True, batch_first=True, init_token = '<trg_sos>', eos_token='<trg_eos>', tokenize=bert )

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC_FIELD, TRG_FIELD))


