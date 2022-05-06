
import torch 
from datasets import load_dataset
from transformers import AutoTokenizer 
# from _config import Config as config 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import translation_utils
from translation_utils import vocab 
import os 


os.environ['TRANSFORMERS_OFFLINE'] = 'yes' 
class Translation_dataset_t(Dataset):
    
    def __init__(self, 
            train: bool = True):
      
        if train: 
            split = "train" 
        else: 
            split = "test" 
        print('getting dataset')
        self.dataset = load_dataset('wmt14', "de-en", split=split) 
        self.de_list = []
        self.en_list = []
#        self.tokenizer = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
        en_list_2 = []
        #for k in range(100):#len(self.dataset)):
        #   n,i = self.dataset[k]
        for n, i in enumerate(self.dataset): 
            en_list_2.append(i['translation']['en'].lower())
            #print(n)
            if n==500:
                break
        print(len(en_list_2))
        # print(max(en_list_2))
        print('error not found') 
        token_res = self.tokenizer(en_list_2, padding='max_length',max_length=512, return_tensors='pt', truncation=True)['input_ids']
        a1 = list(token_res)
        print('error') 
        self.en_vocab, self.en_vocab_size = vocab(a1)
        self.bert2id_dict = translation_utils.bert2id(self.en_vocab)
        self.id2bert_dict = translation_utils.id2bert(self.en_vocab)
        print('e') 
        

        for n, i in enumerate(self.dataset): 
            #if len(i['translation']['de'])> 400:
            #    print(len(i['translation']['de']))

            #elif len(i['translation']['en'])> 400: 
            #    print(len(i['translation']['en']))
            #    print(i['translation']['en'])

            #else: 
               # print(len(i['translation']['de']))
            if len(i['translation']['de'].lower()) > 500: 
                pass
            elif len(i['translation']['en'].lower())>500: 
                pass
           
            self.de_list.append(self.tokenizer(i['translation']['de'].lower(), padding='max_length', return_tensors='pt',max_length=512, truncation=True)["input_ids"])
            self.en_list.append(self.tokenizer(i['translation']['en'].lower(), padding='max_length', return_tensors='pt',max_length=512, truncation=True)["input_ids"])
          #  if n==500:
          #      break
        '''
        for i in self.dataset: 
            self.de_list.append(self.tokenizer(i['translation']['de'].lower(), 
                        padding=True, return_tensors='pt')["input_ids"])
            self.en_list.append(self.tokenizer(i['translation']['en'].lower(),
                        padding=True, return_tensors='pt')["input_ids"])
          '''  
        # en_list_id = []
        # for i in self.dataset: 
        #     en_list_id.append(i['translation']['en'].lower())

        de_list_1 = []
        for n,i in enumerate(self.dataset): 

            if len(i['translation']['de'].lower()) > 500: 
                pass
            elif len(i['translation']['en'].lower())>500: 
                pass
            de_list_1.append(i['translation']['de'].lower())
            #if n==500:
                #break

        a = list(self.tokenizer(de_list_1, padding='max_length', return_tensors='pt',max_length=512, truncation=True)['input_ids'])

        en_list_1 = []
        for n,i in enumerate(self.dataset): 
          en_list_1.append(i['translation']['en'].lower())
          if n==500:
              break

        b = list(self.tokenizer(de_list_1, padding='max_length', max_length=512, return_tensors='pt', truncation=True)['input_ids'])
        # en_vocab, self.en_vocab_size = vocab(b)
        self.de_vocab, self.de_vocab_size = vocab(a) 
            
  
  #should return the length of the dataset  
    def __len__(self): 
        return len(self.de_list)

  #should return a particular example
    def __getitem__(self, index): 
        src = self.de_list[index]
        trg = self.en_list[index]
        
        return {'src':src, 'trg':trg}



class MyCollate:
  def __init__(self, 
          tokenizer, 
          bert2id_dict: dict):
    self.tokenizer = tokenizer
    self.pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
    self.bert2id_dict = bert2id_dict 

  def __call__(self, batch):

    source = []
    for i in batch:
      source.append(i['src'].T)
    #print(source[0].shape, source[1].shape)
    source = pad_sequence(source, batch_first=False, padding_value=self.pad_idx)

    target = []
    for i in batch:
      target.append(i['trg'].T)
    target = pad_sequence(target, batch_first=False, padding_value = self.pad_idx)
    
    target_inp = target.squeeze(-1)[:-1, :]
    target_out = torch.zeros(target.shape)

    for i in range(len(target)): 
        for j in range(len(target[i])): 
            try: 
                target_out[i][j] = self.bert2id_dict[target[i][j].item()]
            except KeyError: 
                target_out[i][j] = self.tokenizer.unk_token_id

    target_out = target_out.squeeze(-1)[1:, :]

    return source.squeeze(), target.squeeze().long(), target_inp.squeeze().long(), target_out.squeeze().long()  


# dataset = Translation_dataset()
# loader = DataLoader(dataset=dataset, 
#                       batch_size= 32, 
#                       shuffle=False,
#                       collate_fn=MyCollate())
