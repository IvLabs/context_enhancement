## Steps to run the model : 

### 1. Install the requirements
```
$ pip install -r requirements.txt
```
### 2. Login to your wandb account: 
```
$ wandb login
```
(This will prompt a link to the wandb auth key. Copy and paste it in the terminal.) 

### 3. Run the main function:
```
$ python main.py
```

### Arguments (Ignore for now. All values are default): 

| argument | help | 
|:--: | :--: |
|--workers | number of data loader workers |
|--batch-size | mini-batch size |
|--learning-rate-weights | base learning rate for weights | 
|--learning-rate-biases | base learning rate for biases |
|--weight-decay| weight-decay |
|--lambd| weight on off-diagonal terms |
|--projector| projector MLP | 
| --print-freq | print frequency |
| --dmodel | dimension of transformer encoder | 
| --nhead | number of heads in transformer | 
| --dfeedforward | dimension of feedforward layer in transformer encoder | 
| --nlayers | number of layers of transformer encoder |
| --tokenizer | tokenizer | 
|--mbert-out-size | Dimenision of mbert output | 
|--checkpoint-dir | path to checkpoint directory | 
