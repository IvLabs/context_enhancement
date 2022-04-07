from datasets import load_dataset
from transformers import AutoTokenizer 
# from _config import Config as config 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

class Translation_dataset(Dataset):
    
    def __init__(self):
      
        self.dataset = load_dataset('opus_rf', "de-en", split="train") 
        self.de_list = []
        self.en_list = []

        for i in self.dataset: 
            self.de_list.append(tokenizer(i['translation']['de'].lower(), padding=True, return_tensors='pt')["input_ids"])
            self.en_list.append(tokenizer(i['translation']['en'].lower(), padding=True, return_tensors='pt')["input_ids"])
            

  
  #should return the length of the dataset  
    def __len__(self): 
        return len(self.de_list)

  #should return a particular example
    def __getitem__(self, index): 
        src = self.de_list[index]
        trg = self.en_list[index]
        
        return {'src':src, 'trg':trg}



class MyCollate:
  def __init__(self):
    self.pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

  def __call__(self, batch):

    source = []
    for i in batch:
      source.append(i['src'].T)
    #print(source[0].shape, source[1].shape)
    source = pad_sequence(source, batch_first=True, padding_value=self.pad_idx)

    target = []
    for i in batch:
      target.append(i['trg'].T)
    target = pad_sequence(target, batch_first=True, padding_value = self.pad_idx)

    return source, target 


# dataset = Translation_dataset()
# loader = DataLoader(dataset=dataset, 
#                       batch_size= 32, 
#                       shuffle=False,
#                       collate_fn=MyCollate())
