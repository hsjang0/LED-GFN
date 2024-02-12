from itertools import chain
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch_scatter import scatter, scatter_sum
import wandb
import torch.nn as nn

from .basegfn import BaseTBGFlowNet, tensor_to_np, unroll_trajs
from .. import network, utils

from torch.distributions import Categorical
from pathlib import Path

from .basegfn import unroll_trajs
from ..data import Experience
from ..utils import tensor_to_np, batch, pack, unpack
from ..network import make_mlp, StateFeaturizeWrap

class Empty(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
  def train(self, batch):
    return


class TBGFN(BaseTBGFlowNet):
  def train(self, batch):
    return self.train_tb(batch)
  

class SubstructureGFN(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)

  def train(self, batch):
    return self.train_substructure(batch)

  def train_substructure(self, batch, log = True):
    fwd_chain = self.batch_traj_fwd_logp(batch)
    back_chain = self.batch_traj_back_logp(batch)

    # 1. Obtain back policy loss
    logp_guide = torch.stack([exp.logp_guide for exp in batch])
    back_losses = torch.square(back_chain - logp_guide)
    back_losses = torch.clamp(back_losses, max=10**2)
    mean_back_loss = torch.mean(back_losses)

    # 2. Obtain TB loss with target: mix back_chain with logp_guide
    targets = []
    for i, exp in enumerate(batch):
      if exp.logp_guide is not None:
        w = self.args.target_mix_backpolicy_weight
        target = w * back_chain[i].detach() + (1 - w) * (exp.logp_guide + exp.logr)
      else:
        target = back_chain[i].detach()
      targets.append(target)
    targets = torch.stack(targets)

    tb_losses = torch.square(fwd_chain - targets)
    tb_losses = torch.clamp(tb_losses, max=10**2)
    loss_tb = torch.mean(tb_losses)

    # 1. Update back policy on back loss
    self.optimizer_back.zero_grad()
    loss_step1 = mean_back_loss
    loss_step1.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_back.step()
    if log:
      loss_step1 = tensor_to_np(loss_step1)

    # 2. Update fwd policy on TB loss
    self.optimizer_fwdZ.zero_grad()
    loss_tb.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_fwdZ.step()
    self.clamp_logZ()
    if log:
      loss_tb = tensor_to_np(loss_tb)

    if log:
      logZ = tensor_to_np(self.logZ)
      wandb.log({
        'Sub back loss': loss_step1,
        'Sub fwdZ loss': loss_tb,
        'Sub logZ': logZ,
      })
    return

  

class SubTBGFN(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    
  def init_subtb(self):
    r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
    \sum_{m=1}^{T-1} \sum_{n=m+1}^T
        \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                    {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
    """
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    # This will contain a sequence of repeated ranges, e.g.
    # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]
    # We need two sets of indices, the first are the source indices, the second the destination
    # indices. We precompute such indices for every possible trajectory length.

    # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
    # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
    # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

    # The destination indices indicate the index of the subsequence the source indices correspond to.
    # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
    # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

    # And so with these indices, for example for m=0, n=3, the forward probability
    # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch):
    self.init_subtb()
    return self.train_subtb(batch)
  
  def train_subtb(self, batch, log = True):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_sub_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return
  
  def batch_loss_sub_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()
      
    total_loss = torch.zeros(len(batch), device=self.args.device)
    ar = torch.arange(self.subtb_max_len)
    for i in range(len(batch)):
      # Luckily, we have a fixed terminal length
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    losses = torch.clamp(total_loss, max=2000)
    mean_loss = torch.mean(losses)
    return mean_loss



class DBGFN(BaseTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    
  def train(self, batch):
    return self.train_db(batch)
  
  def train_db(self, batch, log = True):
    """ Step on detailed balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_detailed_balance(batch)
    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
      
    if log:
      batch_loss = tensor_to_np(batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
      
    del batch_loss
    return
  
  def batch_loss_detailed_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()

    losses = (log_F_s + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    losses = torch.clamp(losses, max=2000)
    mean_loss = torch.mean(losses)
    return mean_loss



class DBGFN_RD(BaseTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )    
    self.count = 0
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    net = network.make_mlp(
        [self.actor.ft_dim*2] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.pR = network.StateFeaturizeWrap_LED(net, self.actor.featurize)
    self.pR.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    self.clip_grad_norm_params.append(self.pR.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      },{
        'params': self.pR.parameters(),
        'lr': 1e-3
      }])
    
    self.optimizers.append(self.optimizer_logF)
    
  def train(self, batch):
    return self.train_db(batch)



  def train_proxy(self, batch):
    for opt in self.optimizers:
      opt.zero_grad()
    
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    inputs = fwd_states + [fwd_states[0]]
    potential = self.pR(inputs)
    potential_2d = torch.zeros((len(batch), self.mdp.forced_stop_len+1)).to(device=self.args.device)

    for traj_idx, (start, end) in unroll_idxs.items():
      for i, j in enumerate(range(start, end)):
        potential_2d[traj_idx][i] = potential[j]
    potential_2d[:,-1] = 0
    potential = potential_2d
    
    losses_est = 0
    for i, exp in enumerate(batch):
      mask_r = torch.rand((len(potential[i,:-1]))).to(self.args.device)
      mask_r[mask_r>=self.args.dropout] = 1
      mask_r[mask_r<self.args.dropout] = 0
      if torch.sum(mask_r) == 0:
        mask_r[-1] = 1
      losses_est += ((len(potential[i,:-1])*torch.sum(potential[i,:-1]*mask_r)/torch.sum(mask_r)
                      - exp.logr.clone().detach())**2)

    losses_est = losses_est / len(batch)
    losses_est = torch.clamp(losses_est, max=2000)
    losses_est.backward()
  
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    return
  
  
  def train_db(self, batch, log = True):
    batch_loss = self.batch_loss_detailed_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
      
    if log:
      batch_loss = tensor_to_np(batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
      
    return
  
  def batch_loss_detailed_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    potential, log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    losses_est = 0
    for i, exp in enumerate(batch):
      mask_r = torch.rand((len(potential[i,:-1]))).to(self.args.device)
      mask_r[mask_r>=self.args.dropout] = 1
      mask_r[mask_r<self.args.dropout] = 0
      if torch.sum(mask_r) == 0:
        mask_r[-1] = 1
      losses_est += ((len(potential[i,:-1])*torch.sum(potential[i,:-1]*mask_r)/torch.sum(mask_r)
                      - exp.logr.clone().detach())**2)
      potential[i,:-1] = (potential[i,:-1] + 
                           (exp.logr.clone().detach()-torch.sum(potential[i,:-1]).detach()
                            )/len(potential[i,:-1])) 
    losses_est = losses_est / len(batch)
    losses_est = torch.clamp(losses_est, max=2000)
    
    potential = potential.detach()
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = 0
    
    losses = (log_F_s - potential + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    losses = torch.clamp(losses, max=2000)
    mean_loss = torch.mean(losses)
    self.count += 1
    return mean_loss + losses_est


  def batch_traj_fwd_logp_unroll(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)
    
    inputs = fwd_states + [fwd_states[0]]
    potential = self.pR(inputs)
    states_to_logfs = self.fwd_logfs_unique(fwd_states)
    states_to_logps = self.fwd_logps_unique(fwd_states)
    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]

    potential_2d = torch.zeros((len(batch), self.mdp.forced_stop_len+1)).to(device=self.args.device)
    log_F_s = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    log_pf_actions = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    for traj_idx, (start, end) in unroll_idxs.items():
      for i, j in enumerate(range(start, end)):
        potential_2d[traj_idx][i] = potential[j]
        log_F_s[traj_idx][i] = states_to_logfs[j]
        log_pf_actions[traj_idx][i] = fwd_logp_chosen[j]

    potential_2d[:,-1] = 0

    return potential_2d, log_F_s, log_pf_actions




class SubTBGFN_RD(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    net = network.make_mlp(
        [self.actor.ft_dim*2] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.pR = network.StateFeaturizeWrap_LED(net, self.actor.featurize)
    self.pR.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    self.clip_grad_norm_params.append(self.pR.parameters())
    
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      },{
        'params': self.pR.parameters(),
        'lr': 1e-3
      }])
    self.count = 0
    self.optimizers.append(self.optimizer_logF)
    
  def init_subtb(self):
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]
    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch):
    self.init_subtb()
    return self.train_subtb(batch)
  
  def train_subtb(self, batch, log = True):
    batch_loss = self.batch_loss_sub_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return

  def train_proxy(self, batch):
    for opt in self.optimizers:
      opt.zero_grad()
    
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    inputs = fwd_states + [fwd_states[0]]
    potential = self.pR(inputs)
    potential_2d = torch.zeros((len(batch), self.mdp.forced_stop_len+1)).to(device=self.args.device)

    for traj_idx, (start, end) in unroll_idxs.items():
      for i, j in enumerate(range(start, end)):
        potential_2d[traj_idx][i] = potential[j]
    potential_2d[:,-1] = 0
    potential = potential_2d
    
    losses_est = 0
    for i, exp in enumerate(batch):
      mask_r = torch.rand((len(potential[i,:-1]))).to(self.args.device)
      mask_r[mask_r>=self.args.dropout] = 1
      mask_r[mask_r<self.args.dropout] = 0
      if torch.sum(mask_r) == 0:
        mask_r[-1] = 1
      losses_est += ((len(potential[i,:-1])*torch.sum(potential[i,:-1]*mask_r)/torch.sum(mask_r)
                      - exp.logr.clone().detach())**2)

    losses_est = losses_est / len(batch)
    losses_est = torch.clamp(losses_est, max=2000)
    losses_est.backward()
  
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    return


  def batch_loss_sub_trajectory_balance(self, batch):
    potential, log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    losses_est = 0
    for i, exp in enumerate(batch):
      mask_r = torch.rand((len(potential[i,:-1]))).to(self.args.device)
      mask_r[mask_r>=self.args.dropout] = 1
      mask_r[mask_r<self.args.dropout] = 0
      if torch.sum(mask_r) == 0:
        mask_r[-1] = 1
      losses_est += ((len(potential[i,:-1])*torch.sum(potential[i,:-1]*mask_r)/torch.sum(mask_r)
                      - exp.logr.clone().detach())**2)
      potential[i,:-1] = (potential[i,:-1] + 
                           (exp.logr.clone().detach()-torch.sum(potential[i,:-1]).detach()
                            )/len(potential[i,:-1])) 
      
    losses_est = losses_est / len(batch)
    losses_est = torch.clamp(losses_est, max=2000)
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = 0
    
    total_loss = torch.zeros(len(batch), device=self.args.device)
    potential = potential.detach()
    for i in range(len(batch)):
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)
      potential_sums = scatter_sum(potential[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - potential_sums - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    losses = torch.clamp(total_loss, max=2000)
    mean_loss = torch.mean(losses)
    self.count += 1
    return mean_loss + losses_est
  
  
  def batch_traj_fwd_logp_unroll(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)
    
    inputs = fwd_states + [fwd_states[0]]
    potential = self.pR(inputs)
    states_to_logfs = self.fwd_logfs_unique(fwd_states)
    states_to_logps = self.fwd_logps_unique(fwd_states)
    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]

    potential_2d = torch.zeros((len(batch), self.mdp.forced_stop_len+1)).to(device=self.args.device)
    log_F_s = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    log_pf_actions = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    for traj_idx, (start, end) in unroll_idxs.items():
      for i, j in enumerate(range(start, end)):
        potential_2d[traj_idx][i] = potential[j]
        log_F_s[traj_idx][i] = states_to_logfs[j]
        log_pf_actions[traj_idx][i] = fwd_logp_chosen[j]
    
    potential_2d[:,-1] = 0    
    return potential_2d, log_F_s, log_pf_actions



class DBGFN_FL(BaseTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )    
    self.train_count = 0
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
        
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    
  def train(self, batch):
    return self.train_db(batch)
  
  def train_db(self, batch, log = True):
    """ Step on detailed balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_detailed_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
      
    if log:
      batch_loss = tensor_to_np(batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
      
    return
  
  def batch_loss_detailed_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    potential, log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    potential = potential.detach()
      
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = 0#exp.logr.clone().detach()

    losses = (log_F_s - potential + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    losses = torch.clamp(losses, max=2000)
    mean_loss = torch.mean(losses)
    
    self.train_count += 1
    return mean_loss


  def batch_traj_fwd_logp_unroll(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logfs = self.fwd_logfs_unique(fwd_states)
    states_to_logps = self.fwd_logps_unique(fwd_states)
    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]

    potential_2d = torch.zeros((len(batch), self.mdp.forced_stop_len+1)).to(device=self.args.device)
    log_F_s = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    log_pf_actions = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    for traj_idx, (start, end) in unroll_idxs.items():
      temp = 0
      for i, j in enumerate(range(start, end)):
        if j == end-1:
          potential_2d[traj_idx][i] = 0
        elif i == 0:
          potential_2d[traj_idx][i] = np.log(self.mdp.reward(fwd_states[j+1]))#-np.log(self.mdp.reward(fwd_states[j]))
        else: 
          potential_2d[traj_idx][i] = np.log(self.mdp.reward(fwd_states[j+1]))-np.log(self.mdp.reward(fwd_states[j]))#.detach()
        log_F_s[traj_idx][i] = states_to_logfs[j]
        log_pf_actions[traj_idx][i] = fwd_logp_chosen[j]
        
    return potential_2d, log_F_s, log_pf_actions



class SubTBGFN_FL(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
  
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    
  def init_subtb(self):
    r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
    \sum_{m=1}^{T-1} \sum_{n=m+1}^T
        \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                    {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
    """
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]

    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch):
    self.init_subtb()
    return self.train_subtb(batch)


  def train_subtb(self, batch, log = True):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_sub_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return
  
  def batch_loss_sub_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    potential, log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = 0
      
    total_loss = torch.zeros(len(batch), device=self.args.device)
    ar = torch.arange(self.subtb_max_len)
    potential = potential.detach()
    for i in range(len(batch)):
      # Luckily, we have a fixed terminal length
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)
      potential_sums = scatter_sum(potential[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - potential_sums - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    losses = torch.clamp(total_loss, max=2000)
    mean_loss = torch.mean(losses)
    return mean_loss
  
  
  def batch_traj_fwd_logp_unroll(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logfs = self.fwd_logfs_unique(fwd_states)
    states_to_logps = self.fwd_logps_unique(fwd_states)
    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]

    potential_2d = torch.zeros((len(batch), self.mdp.forced_stop_len+1)).to(device=self.args.device)
    log_F_s = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    log_pf_actions = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    for traj_idx, (start, end) in unroll_idxs.items():
      temp = 0
      for i, j in enumerate(range(start, end)):
        if j == end-1:
          potential_2d[traj_idx][i] = 0
        elif i == 0:
          potential_2d[traj_idx][i] = np.log(self.mdp.reward(fwd_states[j+1]))#-np.log(self.mdp.reward(fwd_states[j]))
        else:
          potential_2d[traj_idx][i] = np.log(self.mdp.reward(fwd_states[j+1]))-np.log(self.mdp.reward(fwd_states[j]))#.detach()
        log_F_s[traj_idx][i] = states_to_logfs[j]
        log_pf_actions[traj_idx][i] = fwd_logp_chosen[j]
        
    return potential_2d, log_F_s, log_pf_actions




class MaxEntGFN(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)

  def train(self, batch):
    return self.train_tb(batch)

  def back_logps_unique(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_dicts = []
    for state in batch:
      parents = self.mdp.get_unique_parents(state)
      logps = np.log([1/len(parents) for parent in parents])

      state_to_logp = {parent: logp for parent, logp in zip(parents, logps)}
      batch_dicts.append(state_to_logp)
    return batch_dicts if batched else batch_dicts[0]

  def back_sample(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_samples = []
    for state in batch:
      sample = np.random.choice(self.mdp.get_unique_parents(state))
      batch_samples.append(sample)
    return batch_samples if batched else batch_samples[0]



class PPO():
    """
        Proximal Policy Gradient
        Actor: SSR style
        Critic: SA style
    """
    
    def __init__(self, args, mdp, actor):
        self.args = args
        self.mdp = mdp
        self.actor = actor
        
        self.policy = actor.policy_fwd
        self.policy_back = actor.policy_back # not used
        
        hid_dim = self.args.sa_hid_dim
        n_layers = self.args.sa_n_layers
        net = make_mlp(
            [self.actor.ft_dim] + \
            [hid_dim] * n_layers + \
            [1]
        )
        self.critic = StateFeaturizeWrap(net, self.actor.featurize)
        self.critic.to(args.device)
        
        self.nets = [self.policy, self.critic]
        for net in self.nets:
            net.to(self.args.device)
            
        self.clip_grad_norm_params = [self.policy.parameters(),
                                      self.critic.parameters()]
        
        self.optimizer = torch.optim.Adam([
            {
                'params': self.policy.parameters(),
                'lr': args.lr_policy
            }, {
                'params': self.critic.parameters(),
                'lr': args.lr_critic
            }
        ])
    
    def fwd_sample(self, batch, epsilon=0.0):
        return self.policy.sample(batch, epsilon=epsilon)
    
    def fwd_logps_unique(self, batch):
        return self.policy.logps_unique(batch)
    
    def batch_fwd_sample(self, n, epsilon=0.0, uniform=False):
        """ Batch samples dataset with n items.

            Parameters
            ----------
            n: int, size of dataset.
            epsilon: Chance in [0, 1] of uniformly sampling a unique child.
            uniform: If true, overrides epsilon to 1.0
            unique: bool, whether all samples should be unique

            Returns
            -------
            dataset: List of [Experience]
        """
        if uniform:
            epsilon = 1.0
        incomplete_trajs = [[self.mdp.root()] for _ in range(n)]
        complete_trajs = []
        logps_trajs = [[] for _ in range(n)]
        while len(incomplete_trajs) > 0:
            inp = [t[-1] for t in incomplete_trajs]
            samples = self.fwd_sample(inp, epsilon=epsilon)
            logps = self.fwd_logps_unique(inp)
            for i, (logp, sample) in enumerate(zip(logps, samples)):
                incomplete_trajs[i].append(sample)
                logps_trajs[i].append(logp[sample].cpu().detach())
        
            # Remove complete trajs that hit leaf
            temp_incomplete = []
            for t in incomplete_trajs:
                if not t[-1].is_leaf:
                    temp_incomplete.append(t)
                else:
                    complete_trajs.append(t)
            incomplete_trajs = temp_incomplete

        # convert trajs to exps
        list_exps = []
        for traj, logps_traj in zip(complete_trajs, logps_trajs):
            x = traj[-1]
            r = self.mdp.reward(x)
            # prevent NaN
            exp = Experience(traj=traj, x=x, r=r,
                logr=torch.nan_to_num(torch.log(torch.tensor(r, dtype=torch.float32)).to(device=self.args.device), neginf=-100.0),
                logp_guide=logps_traj)
            list_exps.append(exp)
        return list_exps
        
    def train(self, batch, log = True):
        batch_loss = self.batch_loss_ppo(batch)
        
        self.optimizer.zero_grad()
        batch_loss.backward()
        
        for param_set in self.clip_grad_norm_params:
            torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
        self.optimizer.step()
        
        if log:
            batch_loss = tensor_to_np(batch_loss)
            wandb.log({'PPO loss': batch_loss})
            return
        
    def batch_loss_ppo(self, batch):
        trajs = [exp.traj for exp in batch]
        old_fwd_logp_chosen = [exp.logp_guide for exp in batch]
        fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)
        
        states_to_logps = self.fwd_logps_unique(fwd_states)
        fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]
        old_log_probs = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        log_probs = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, (old_log_prob, log_prob) in enumerate(zip(old_fwd_logp_chosen, fwd_logp_chosen)):
            old_log_probs[i] = old_log_prob[i % (self.mdp.forced_stop_len + 1)].to(self.args.device)
            log_probs[i] = log_prob
            
        old_log_probs = self.clip_policy_logits(old_log_probs)
        old_log_probs = torch.nan_to_num(old_log_probs, neginf=self.args.clip_policy_logit_min)
        
        log_probs = self.clip_policy_logits(log_probs)
        log_probs = torch.nan_to_num(log_probs, neginf=self.args.clip_policy_logit_min)

        V = self.critic(fwd_states)
        # The return is the terminal reward everywhere, we're using gamma==1
        G = torch.FloatTensor([exp.r for exp in batch]).repeat_interleave(self.mdp.forced_stop_len + 1).to(self.args.device)
        A = G - V
        
        V_loss = A.pow(2)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * A
        pol_objective = torch.min(surr1, surr2)
        entropy = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, s2lp in enumerate(states_to_logps):
            for state, logp in s2lp.items():
                entropy[i] = -torch.sum(torch.exp(logp) * logp)
        pol_objective = pol_objective + self.args.entropy_coef * entropy
        pol_loss = -pol_objective
        
        loss = V_loss + pol_loss
        loss = torch.clamp(loss, max=5000)
        mean_loss = torch.mean(loss)
        return mean_loss
    
    def save_params(self, file):
        Path('/'.join(file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        #torch.save({
        #    'policy':   self.policy.state_dict(),
        #    'critic':  self.critic.state_dict(),
        #    }, file)
        return

    def load_for_eval_from_checkpoint(self, file):
        checkpoint = torch.load(file)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        for net in self.nets:
            net.eval()
        return

    def clip_policy_logits(self, scores):
        return torch.clip(scores, min=self.args.clip_policy_logit_min,
                                  max=self.args.clip_policy_logit_max)









def make_model(args, mdp, actor):
  """ Constructs MaxEnt / TB / Sub GFN. """
  if args.model == 'maxent':
    model = MaxEntGFN(args, mdp, actor)
  elif args.model == 'tb':
    model = TBGFN(args, mdp, actor)
  elif args.model == 'sub':
    model = SubstructureGFN(args, mdp, actor)
  elif args.model == "subtb":
    model = SubTBGFN(args, mdp, actor)
  elif args.model == "subtb_rd":
    model = SubTBGFN_RD(args, mdp, actor)
  elif args.model == "subtb_fl":
    model = SubTBGFN_FL(args, mdp, actor)
  elif args.model == 'db':
    model = DBGFN(args, mdp, actor)
  elif args.model == 'db_rd':
    model = DBGFN_RD(args, mdp, actor)
  elif args.model == 'db_fl':
    model = DBGFN_FL(args, mdp, actor)
  elif args.model == 'random':
    args.explore_epsilon = 1.0
    args.num_offline_batches_per_round = 0
    model = Empty(args, mdp, actor)
  elif args.model == 'ppo':
    model = PPO(args, mdp, actor)
  return model







