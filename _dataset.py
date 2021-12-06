import datasets
from datasets import  load_dataset
from _config import Config as config
from transformers import AutoTokenizer 
from transformers import BertModel

# Tokenizing and Making the Iterator 
## initializing the tokenizer: 
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER) 

class Dataset: 
    def __init__(self, config):
        # Loading Dataset 
        dataset = load_dataset('wmt14', config.LANG_PAIR, split=config.SPLIT) 

        ## making a list of de and en sentences: 
        de_list = [] 
        en_list = [] 
        for i in dataset: 
            de_list.append(i['translation']['de'].lower())
            en_list.append(i['translation']['en'].lower())

        ## making the iterator 
        iterator  = {'de': [], 'en': []}
        batch_size = config.BATCH_SIZE 
        for i in range(len(de_list)//batch_size): 
            de_batch = de_list[i*(batch_size):(i+1)*batch_size]
            en_batch = en_list[i*batch_size: (i+1)*batch_size]
            # print(en_batch[1])

            iterator["de"].append(tokenizer(de_batch, padding=True, return_tensors="pt")["input_ids"])
            iterator["en"].append(tokenizer(en_batch, padding=True, return_tensors="pt")["input_ids"])
            # print(test_iter["de"][0][1])

        de_batch = de_list[(i+1)*batch_size:]
        en_batch = en_list[(i+1)*batch_size:]
            
        iterator["en"].append(tokenizer(en_batch, padding=True, return_tensors="pt")["input_ids"])
        iterator["de"].append(tokenizer(de_batch, padding=True, return_tensors="pt")["input_ids"])

        self.iterator = iterator