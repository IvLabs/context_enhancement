# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
from translation_dataset import Translation_dataset
from translation_dataset import MyCollate
from transformers import BertModel
from transformers import AutoTokenizer
from torch import nn, optim
import torch

import wandb 
#from _config import Config 
#config = Config.config

os.environ['WANDB_START_METHOD'] = 'thread'

#setting random seeds
SEED = 4444

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True




parser = argparse.ArgumentParser(description='Barlow Twins Training')
# parser.add_argument('data', type=Path, metavar='DIR',
#                     help='path to dataset')



# Training parameters: 
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--clip', default=1, type=float, metavar='GC',
                    help='Gradient Clipping')

# Model parameters:
parser.add_argument('--projector', default='768-768', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')

# Transformer parameters: 
parser.add_argument('--dmodel', default=768, type=int, metavar='T', 
                    help='dimension of transformer encoder')
parser.add_argument('--nhead', default=3, type= int, metavar='N', 
                    help= 'number of heads in transformer') 
parser.add_argument('--dfeedforward', default=256, type=int, metavar='F', 
                    help= 'dimension of feedforward layer in transformer encoder') 
parser.add_argument('--nlayers', default=3, type=int, metavar= 'N', 
                   help='number of layers of transformer encoder') 

# Tokenizer: 
parser.add_argument('--tokenizer', default='bert-base-multilingual-cased', type=str, 
                metavar='T', help= 'tokenizer')
parser.add_argument('--mbert-out-size', default=768, type=int, metavar='MO', 
                    help='Dimension of mbert output')
# Paths: 
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main():

    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        wandb.init(config=args)#############################################
        # wandb.config.update(args)
        config = wandb.config
        print(args.lambd,config.lambd)
        # wandb.finish()
        # exit()
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    transformer1 = nn.TransformerEncoderLayer(d_model = args.dmodel, nhead=args.nhead, dim_feedforward=args.dfeedforward, batch_first=True)
    t_enc = nn.TransformerEncoder(transformer1, num_layers=args.nlayers)
    model = BarlowTwins(projector_layers=args.projector, mbert_out_size=args.mbert_out_size, transformer_enc=t_enc, lambd=args.lambd).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    # automatically resume from checkpoint if it exists
    # if (args.checkpoint_dir / 'checkpoint.pth').is_file():
    #     ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
    #                       map_location='cpu')
    #     start_epoch = ckpt['epoch']
    #     # print("model=",model)
    #     # print("ckpt=",ckpt['model'])
    #     model.load_state_dict(ckpt['model'])
    #     optimizer.load_state_dict(ckpt['optimizer'])
    # else:
    start_epoch = 0

    ################################
    # dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    ###############################

    dataset = Translation_dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    ###############################
    loader = torch.utils.data.DataLoader(
         dataset, batch_size=per_device_batch_size, num_workers=args.workers,
         pin_memory=True, sampler=sampler, collate_fn = MyCollate(), sampler=sampler)
    #############################
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0 
        for step, (sent) in enumerate(loader, start=epoch * len(loader)):
            y1 = sent[0].cuda(gpu, non_blocking=True)
            y2 = sent[1].cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
                wandb.log({'iter_loss':loss})
#               print(loss.item())
                epoch_loss += loss.item()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        wandb.log({"epoch_loss":epoch_loss})
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    wandb.finish()
    # if args.rank == 0:
    #     # save final model
    #     torch.save(model.module.backbone.state_dict(),
    #                args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
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

        self.mbert = BertModel.from_pretrained(args.tokenizer)

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
        x = self.transformer_enc(x["last_hidden_state"]) 
        x = torch.sum(x, dim=1)/x.shape[1] # using avg pooling 
        
        y = y.squeeze(-1)
        y = self.mbert(y)
        y = self.transformer_enc(y["last_hidden_state"]) 
        y = torch.sum(y, dim=1)/y.shape[1] # using avg pooling 

        x = self.projector(x) #x = [batch_size, projector]
        x = self.bn(x)
        print(x.shape)
        y = self.projector(y)
        y = self.bn(y) #y = [batch_size, projector]

        #emperical cross-correlation mattrix 
        c = x.T @ y

        # for multi-gpu: sum cross correlation matrix between all gpus 
        #(uncomment below 2 lines      
        '''
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)
        
        '''
 
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss 


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


if __name__ == '__main__':
    try:  
      main()
    except KeyboardInterrupt:
      print('Interrupted')
      wandb.finish()
      try:
          sys.exit(0)
      except SystemExit:
          os._exit(0)
