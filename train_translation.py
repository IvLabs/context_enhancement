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

import wandb 


#import barlow
os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
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

def main(): 

    # print("entered main")
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

#        wandb.init(config=args, project='translation_test')#############################################
#        wandb.config.update(args)
#        config = wandb.config
    
        # exit()
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    dataset = Translation_dataset_t(train=args.train) 
    src_vocab_size = dataset.de_vocab_size
    trg_vocab_size = dataset.en_vocab_size
    tokenizer = dataset.tokenizer  
    pad_idx = tokenizer.pad_token_id
    sos_idx = tokenizer.cls_token_id 
    eos_idx = tokenizer.sep_token_id

#    transformer1 = nn.TransformerEncoderLayer(d_model = args.dmodel, nhead=args.nhead, dim_feedforward=args.dfeedforward, batch_first=True)
    # t_enc = nn.TransformerEncoder(transformer1, num_layers=args.nlayers)
    # print(src_vocab_size, trg_vocab_size)
    mbert = BertModel.from_pretrained('bert-base-multilingual-uncased')
    transformer = Transformer(d_model=args.dmodel, 
                              nhead=args.nhead, 
                              num_encoder_layers=args.nlayers, 
                              num_decoder_layers = args.nlayers, 
                              dim_feedforward=args.dfeedforward, 
                              dropout=args.dropout)
    model = Translator(mbert=mbert, transformer= transformer, tgt_vocab_size=trg_vocab_size, emb_size=args.mbert_out_size).cuda(gpu)
    # print(model.state_dict)
#    model_barlow = barlow.BarlowTwins(projector_layers=args.projector, mbert_out_size=args.mbert_out_size, transformer_enc=model.transformer.encoder, lambd=args.lambd).cuda(gpu)

    # args.load = False

    if args.load == 1: 
        # print(args.load)
        # print('inside')
        print('loading barlow model')
        t_enc = model.transformer.encoder
        barlow = BarlowTwins(projector_layers=args.projector, mbert_out_size=args.mbert_out_size, transformer_enc=t_enc, mbert=mbert, lambd=0.0051).cuda(gpu)
        ### note: lambd is just a placeholder
        ckpt = torch.load(args.checkpoint_dir/ 'barlow_checkpoint.pth', 
                            map_location='cpu')
        barlow.load_state_dict(ckpt['model'])
        model.transformer.encoder = barlow.transformer_enc        
        model.mbert = barlow.mbert
    '''
    to_do: 
    if post_train: 
        torch.load(model.states_dict)
        model.transformer.encoder = model_barlow

    '''
#    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

###########################################################
    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps) 
    
    if args.loss_fn == 'cross_entropy': 
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
##############################################################

    start_epoch = 0 

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    ###############################
    loader = torch.utils.data.DataLoader(
         dataset, batch_size=per_device_batch_size, num_workers=args.workers,
         pin_memory=True, sampler=sampler, collate_fn = MyCollate(tokenizer=tokenizer,bert2id_dict=dataset.bert2id_dict))
   
    test_loader = torch.utils.data.DataLoader(
         dataset, batch_size=1, num_workers=args.workers,
         pin_memory=True, sampler=sampler, collate_fn = MyCollate(tokenizer=tokenizer,bert2id_dict=dataset.bert2id_dict))
    #############################
    start_time = time.time()


    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0 
        for step, (sent) in enumerate(loader, start=epoch * len(loader)):
            src = sent[0].cuda(gpu, non_blocking=True)
            tgt_inp = sent[2].cuda(gpu, non_blocking=True)
            tgt_out = sent[3].cuda(gpu, non_blocking=True)
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_inp, pad_idx) 
            logits = model(src, tgt_inp, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            optimizer.zero_grad()

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            # losses += loss.item()
            
#            wandb.log({'iter_loss': loss})
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        # wandb.log({"epoch_loss":epoch_loss})
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.module.state_dict(),
                         optimizer=optimizer.state_dict())
            # print(model.state_dict)
            torch.save(state, args.checkpoint_dir / 'translation_checkpoint.pth')
            print('translation model saved in', args.checkpoint_dir)
        
##############################################################
        if epoch%args.checkbleu ==0 : 

            model.eval()
            predicted=[]
            target=[]
            
            for i in test_loader: 
                src = i[0].cuda(gpu, non_blocking=True)
                tgt_out = i[3].cuda(gpu, non_blocking=True)
                num_tokens = src.shape[0]

                src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).cuda(gpu, non_blocking=True)
                out = translate(model, src, tokenizer, src_mask, gpu)
                predicted.append(out)
                target.append([tokenizer.convert_ids_to_tokens(tgt_out)])
                
                try: 
                    bleu_score(predicted, target)
                except: 
                    predicted.pop()
                    target.pop()
            
            print(bleu_score(predicted, target))
##############################################################
#        if epoch%1 ==0 : 
#            torch.save(model.module.state_dict(),
#                   'path.pth')
#            print("Model is saved")
        # if args.rank == 0:
        #     # save checkpoint
        #     state = dict(epoch=epoch + 1, model=model.state_dict(),
        #                  optimizer=optimizer.state_dict())
        #     torch.save(state, args.checkpoint_dir / f'translation_checkpoint.pth')
        #     print('saved translation model in', args.checkpoint_dir)
#    wandb.finish()
           


'''
todo: 
    BLEU score
'''

# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_idx, gpu):
    src = src
    src_mask = src_mask

    memory = model.module.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).cuda(gpu, non_blocking=True)
    for i in range(max_len-1):
        memory = memory
        tgt_mask = (translation_utils.generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).cuda(gpu, non_blocking=True)
        out = model.module.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.module.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, 
        src: torch.tensor, 
        tokenizer,src_mask, gpu):
    model.eval()
    
    num_tokens = src.shape[0]
    
    
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=tokenizer.cls_token_id, eos_idx=tokenizer.sep_token_id, gpu=gpu).flatten()
    return tokenizer.convert_ids_to_tokens(tgt_tokens) 


if __name__ == '__main__': 
    main()
