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

# importing hugging face dataset library 
import datasets
from datasets import load_dataset

# importing Huggingface BERT model 
from transformers import AutoTokenizer, BertModel

'''' CONFIG PARAMS :

BATCH_SIZE
D_MODEL 
D_FORWARD
NUM_LAYERS
NUM_HEADS
PROJECTOR

'''

parser = argparse.ArgumentParser(description='Barlow Twins Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
########### Adding args ########################
parser.add_argument('--d_model', default=768, type=int, metavar='T', help='dimension of the transformer encoder')
parser.add_argument('--num-heads', default=6, type=int, metavar= 'T', help= 'number of heads in transformer encoder') 
parser.add_argument('--num-layers', default=8, type=int, metavar='T', help= 'number of heads in transformer encoder') 
parser.add_argument('--d-forward', default=1024, type=int, metavar='T', help= 'dimension of transformer  feedforword layer')

## literally from barlow_twins : #############################3

def main():
    args = parser.parse_args()
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
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
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
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')


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
################################################################################

# hugging face mbert tokenizer 
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# fetching dataset
dataset = load_dataset('wmt14', 'de-en', split='test')

# list of dc.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)e and en sentences 
de_list = []
en_list = []
for i in dataset: 
    de_list.append(i['translation']['de'].lower())
    en_list.append(i['translation']['en'].lower())

# making batches of sentences 
test_iter = {'de': [], 'en': []}

BATCH_SIZE = 32

for i in range(len(de_list)//BATCH_SIZE): 
    de_batch = de_list[i*(BATCH_SIZE):(i+1)*BATCH_SIZE]
    en_batch = en_list[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
    # print(en_batch[1])

    test_iter["de"].append(tokenizer(de_batch, padding=True, return_tensors="pt")["input_ids"])
    test_iter["en"].append(tokenizer(en_batch, padding=True, return_tensors="pt")["input_ids"])
    # print(test_iter["de"][0][1])

de_batch = de_list[(i+1)*BATCH_SIZE:]
en_batch = en_list[(i+1)*BATCH_SIZE:]
    
test_iter["en"].append(tokenizer(en_batch, padding=True, return_tensors="pt")["input_ids"])
test_iter["de"].append(tokenizer(de_batch, padding=True, return_tensors="pt")["input_ids"])

# instantiating transformer encoder : 

# defining the model : 
class BarlowTwins(nn.Module):

    def __init__(self, args):   
        super().__init__()
        self.projector_layers = projector_layers
        self.mbert_out_size = mbert_out_size
        self.transformer = nn.TransformerEncoderLayer(d_model=args.d-model, nhead=args.num-heads, dim_feedforward=args.d-forward , batch_first=True)
        self.t_enc = nn.TransformerEncoder(transformer, num_layers=args.num-layer)

        self.mbert = BertModel.from_pretrained("bert-base-multilingual-cased")

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


        x = self.mbert(x)
        x = self.t_enc(x["last_hidden_state"]) #considering sentence representation as the starting special token
        x = torch.sum(x, dim=1)/x.shape[1]

        y = self.mbert(y)
        y = self.t_enc(y["last_hidden_state"]) #considering sentence representation as the starting special token
        y = torch.sum(y, dim=1)/y.shape[1]

        x = self.bn(self.projector(x)) #x = [batch_size, projector]
        y = self.bn(self.projector(y)) #y = [batch_size, projector]

        #emperical cross-correlation mattrix 
        c = x.T @ y

        # for multi-gpu: sum cross correlation matrix between all gpus 
        #(uncomment below 2 lines    
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

 
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag

        return loss 
## Again: quite literally from Barlow Twins: ######################

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
##############################
if __name__ =='__main__': 
    main() 
