import argparse
import gzip
import os
import pdb
import pickle
import threading
import time
import warnings
from copy import deepcopy
import wandb

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import ray

import model_atom, model_block, model_fingerprint
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from data import Dataset, DatasetDirect, get_mol_path_graph, make_model
from proxy import Proxy
from torch.distributions.categorical import Categorical
from utils import chem

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import QED

import copy
from tqdm import tqdm
import datetime
import sys, os

warnings.filterwarnings('ignore')

tmp_dir = "/tmp/molexp"
os.makedirs(tmp_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=4, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.999, type=float)
parser.add_argument("--opt_epsilon", default=1e-8, type=float)
parser.add_argument("--nemb", default=256, help="#hidden", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=8, type=int)
parser.add_argument("--num_iterations", default=250000, type=int)
parser.add_argument("--num_conv_steps", default=3, type=int)
parser.add_argument("--log_reg_c", default=(1/8)**4, type=float)
parser.add_argument("--reward_exp", default=10, type=float)
parser.add_argument("--reward_norm", default=8, type=float)
parser.add_argument("--sample_prob", default=1, type=float)
parser.add_argument("--R_min", default=0.1, type=float)
parser.add_argument("--leaf_coef", default=10, type=float)
parser.add_argument("--clip_grad", default=0.0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
parser.add_argument("--replay_mode", default='online', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--random_action_prob", default=0.05, type=float)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='block_graph')
parser.add_argument("--model_version", default='v4')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--save_path", default='results/')
parser.add_argument("--proxy_path", default='./data/pretrained_proxy')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--progress", default='yes')
parser.add_argument("--floatX", default='float64')
parser.add_argument("--include_nblocks", default=False)
parser.add_argument("--balanced_loss", default=True)
parser.add_argument("--early_stop_reg", default=0.0, type=float)
parser.add_argument("--initial_log_Z", default=30, type=float)
parser.add_argument("--objective", default='subTB', type=str)
# If True this basically implements Buesing et al's TreeSample Q/SoftQLearning, samples uniformly from it though, no MCTS involved
parser.add_argument("--ignore_parents", default=False)
parser.add_argument("--fl", default=0, type=int)
parser.add_argument("--num_samples", default=100, type=int)
parser.add_argument("--decompose_step", default=3, type=int)
parser.add_argument("--dropout_prob", default=0.1, type=float)

#@torch.jit.script
def led_subtb_loss(P_F, P_B, F, R, traj_lengths,transition_rs,Lambda=0.9):        
    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=traj_lengths.device), traj_lengths]), 0).long()
    total_loss = torch.zeros(1, device=traj_lengths.device)
    total_Lambda = torch.zeros(1, device=traj_lengths.device)
    for ep in range(traj_lengths.shape[0]):
        offset = cumul_lens[ep]
        T = int(traj_lengths[ep])
        transition_rs[offset:offset + T-1] += (R[ep] - torch.sum(transition_rs[offset:offset+T-1]))/(T-1)
        for i in range(T):
            for j in range(i, T):
                # This flag is False if the endpoint flow of this subtrajectory is R == F(s_T)
                flag = float(j + 1 < T)
                acc = F[offset + i] - F[offset + min(j + 1, T - 1)] 
                for k in range(i, j + 1):
                    flag = float(k + 1 < T)
                    acc += P_F[offset + k] - P_B[offset + k]  - transition_rs[offset + min(k, T - 1)]*flag
                total_loss += acc.pow(2) * Lambda ** (j - i + 1)
                total_Lambda += Lambda ** (j - i + 1)
    return total_loss / total_Lambda


#@torch.jit.script
def led_db_loss(P_F, P_B, F, R, traj_lengths, transition_rs):
    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=traj_lengths.device), traj_lengths]), 0).long()
    total_loss = torch.zeros(1, device=traj_lengths.device)
    for ep in range(traj_lengths.shape[0]):
        offset = cumul_lens[ep]
        T = int(traj_lengths[ep])
        transition_rs[offset:offset + T-1] += (R[ep] - torch.sum(transition_rs[offset:offset+T-1]))/(T-1)
        
        for i in range(T):
            flag = float(i + 1 < T)

            curr_PF = P_F[offset + i]
            curr_PB = P_B[offset + i]
            curr_F = F[offset + i]
            curr_F_next = F[offset + min(i + 1, T - 1)]
            curr_r = flag*transition_rs[offset + min(i, T - 1)]
            acc = curr_F + curr_PF - curr_F_next - curr_PB - curr_r

            total_loss += acc.pow(2)

    return total_loss

#@torch.jit.script
def learning_decomposition(R_est, R, traj_lengths, dropout_prob):
    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=traj_lengths.device), traj_lengths]), 0).long()
    total_loss = torch.zeros(1, device=traj_lengths.device)
    for ep in range(traj_lengths.shape[0]):
        R_cum = torch.zeros(1, device=traj_lengths.device)
        offset = cumul_lens[ep]
        T = int(traj_lengths[ep])

        mask_r = torch.rand((T-1)).to(R_cum.device)
        mask_r[mask_r>=dropout_prob] = 1
        mask_r[mask_r<dropout_prob] = 0
        mask_r[-1] = 1 if torch.sum(mask_r)==0 else mask_r[-1]

        R_cum = torch.sum(R_est[offset:offset+T-1] * mask_r)
        R_cum = R_cum * (T-1)/torch.sum(mask_r)
        total_loss += ((R_cum - R[ep]).pow(2))
        
    return total_loss / traj_lengths.shape[0]




_stop = [None]

def train_model_with_proxy(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = True
    device = torch.device('cuda')
    dock_pool = proxy


    if num_steps is None:
        num_steps = args.num_iterations + 1
    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(
            model
        )
    if do_save:
        args.run = datetime.datetime.now().strftime('%Y%m%dT%H%M%S.%f')
        exp_dir = f'{args.save_path}/{args.run}/'
        print ('\033[32mexp_dir: {}\033[0m'.format(exp_dir))
        if os.path.exists(exp_dir):
            raise RuntimeError('{} exists'.format(exp_dir))
        else:
            os.makedirs(exp_dir)
        with open(os.path.join(exp_dir[:-1], 'command.txt'), 'w') as f:
            argv = sys.argv
            f.write(' '.join(argv))


    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)
    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()

    opt = torch.optim.Adam(
        model.parameters(), 
        args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(args.opt_beta, args.opt_beta2),
        eps=args.opt_epsilon
    )

    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    loginf = 1000 # to prevent nans
    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])
    balanced_loss = args.balanced_loss
    do_nblocks_reg = False
    max_blocks = args.max_blocks
    leaf_coef = args.leaf_coef



    potential_function = model_block.GraphAgent_rwd(
            nemb=args.nemb,
            nvec=0,
            out_per_stem=dataset.mdp.num_blocks,
            out_per_mol=1,
            num_conv_steps=args.num_conv_steps,
            mdp_cfg=dataset.mdp,
            version=args.model_version,
        ).cuda()

    opt_est = torch.optim.Adam(
        potential_function.parameters(), 
        1e-3, 
        weight_decay=args.weight_decay,
        betas=(args.opt_beta, args.opt_beta2),
        eps=args.opt_epsilon
    )

    for i in range(num_steps):
        if not debug_no_threads:
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            minibatch = r
        else:
            minibatch = dataset.sample2batch(
                dataset.sample(mbsize)
            )
            
        s, a, r, d, n, mols, idc, lens, *o = minibatch
        tzeros = torch.zeros(idc[-1] + 1, device=device, dtype=args.floatX)
        traj_r = tzeros.index_add(0, idc, r)


        # Learning energy decomposition
        opt.zero_grad()
        opt_est.zero_grad()
        for gg in range(0,args.decompose_step):
            potentials = potential_function(s.cuda(), None).view(-1)
            loss = learning_decomposition(potentials, torch.log(traj_r), lens, args.dropout_prob)
            potentials_use = potentials.detach()
            opt_est.zero_grad()
            loss.backward()
            opt_est.step()
        

        stem_out_s, mol_out_s = model(s, None)
        logits = -model.action_negloglikelihood(s, a, 0, stem_out_s, mol_out_s)
        if args.objective == 'subTB':
            loss = led_subtb_loss(logits, torch.log(1/n), mol_out_s[:, 1], torch.log(traj_r), lens, transition_rs=potentials_use)
        elif args.objective == 'detbal':
            loss = led_db_loss(logits, torch.log(1/n), mol_out_s[:, 1], torch.log(traj_r), lens, transition_rs=potentials_use)
        loss.backward()


        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
        opt.step()
        
        
        model.training_steps = i + 1
        if tau > 0:
            for _a,b in zip(model.parameters(), target_model.parameters()):
                b.data.mul_(1 - tau).add_(tau * _a)

        # logging and saving
        if (not i % 100) and i >= 0:
            print({
                "top_100": np.mean(dataset.top_100_rwd()),
                "mode": dataset.modes,
                "similarity": dataset.tasimoto(),
            })






def main(args):
    bpath = "data/blocks_PDB_105.json"
    device = torch.device('cuda')

    if args.floatX == 'float32':
        args.floatX = torch.float
    else:
        args.floatX = torch.double
        
    args.ignore_parents = True
    dataset = DatasetDirect(args, bpath, device, floatX=args.floatX)

    mdp = dataset.mdp

    model = make_model(args, mdp, out_per_mol=1 + (1 if (args.objective in ['detbal','subTB']) else 0))
    model.to(args.floatX)
    model.to(device)

    proxy = Proxy(args, bpath, device)

    train_model_with_proxy(args, model, proxy, dataset, do_save=True)

if __name__ == '__main__':
  args = parser.parse_args()

  if 0:
    all_hps = eval(args.array)(args)
    for run in range(len(all_hps)):
      args.run = run
      hps = all_hps[run]
      for k,v in hps.items():
        setattr(args, k, v)
      exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
      #if os.path.exists(exp_dir):
      #  continue
      print(hps)
      main(args)
  elif args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
    else:
      hps = all_hps[args.run]
      print(hps)
      for k,v in hps.items():
        setattr(args, k, v)
    try:
        main(args)
    except KeyboardInterrupt as e:
        print("stopping for", e)
        _stop[0]()
        raise e
    except Exception as e:
        print("exception", e)
        _stop[0]()
        raise e
  else:
      try:
          main(args)
      except KeyboardInterrupt as e:
          print("stopping for", e)
          _stop[0]()
          raise e
      except Exception as e:
          print("exception", e)
          _stop[0]()
          raise e
