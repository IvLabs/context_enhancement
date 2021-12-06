import torch 
import os 
import torch.nn as nn 
import torch.distributed as dist 
from _config import Config as config 
from _utils import LARS 
from _dataset import Dataset
from _model import Model 

def train(model):

    dataset = Dataset(config=config) 
    iterator = dataset.iterator

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    
    optimizer = LARS(parameters, lr=config.LR, weight_decay=config.WEIGHT_DECAY,
                    momentum=config.MOMENTUM,
                    eta=config.ETA,
                    weight_decay_filter=True,
                    lars_adaptation_filter=True)

    scaler = torch.cuda.amp.GradScaler()
    zipped = list(zip(iterator['en'], iterator['de']))

    for epoch in range(config.EPOCHS):

        epoch_loss = 0 
        for i in zip(iterator['en'], iterator['de']): 
                x = torch.stack(list(i[1]))
                y = torch.stack(list(i[0]))
                # x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = model(x, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                print(loss.item())
                epoch_loss+=loss.item()

        print(epoch_loss)


def main(): 

    transformer1 = nn.TransformerEncoderLayer(d_model=config.D_MODEL, nhead=config.N_HEAD, dim_feedforward=config.DIM_FEEDFORWARD, batch_first=True)
    t_enc = nn.TransformerEncoder(transformer1, num_layers=config.NUM_LAYERS)
    model = Model(projector_layers= config.PROJECTOR_LAYERS, mbert_out_size=config.MBERT_OUT_SIZE, transformer_enc= t_enc, lambd= config.LAMBD)
    train(model = model)

# if __name__ == main: 
#     main()
main()
