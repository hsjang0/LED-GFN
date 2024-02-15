'''
  TFBind8
  Oracle
  Start from scratch
  No proxy
'''
import copy, pickle
import numpy as np
import pandas as pd
import torch
from polyleven import levenshtein

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqinsertmdp, seqarmdp
from gflownet.monitor import TargetRewardDistribution, Monitor

from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.oracles.tensorflow import TransformerOracle

def dynamic_inherit_mdp(base, args):

  class TFBind8MDP(base):
    def __init__(self, args):
      super().__init__(args,
                       alphabet=list('0123'),
                       forced_stop_len=8)
      self.args = args
      args.alphabet = list('0123')
      # Read from file
      print(f'Loading data ...')
      with open('datasets/tfbind8/tfbind8-exact-v0-all.pkl', 'rb') as f:
        oracle_d = pickle.load(f)
      
      munge = lambda x: ''.join([str(c) for c in list(x)])

      self.oracle = {self.state(munge(x), is_leaf=True): float(y)
          for x, y in zip(oracle_d['x'], oracle_d['y'])}
      #print(self.oracle)

      #dataset = TFBind8Dataset()
      #self.proxy_model = TransformerOracle(dataset, noise_std=0.01)
      
      #for x, y in zip(oracle_d['x'], oracle_d['y']):
      #  print(x,y,self.proxy_model.params["model"].predict({"input_ids": np.array([self.char_to_idx[c] for c in self.state(munge(x), is_leaf=True).content]).reshape(1, -1)})[0].item())
      
      # Scale rewards
      self.scaled_oracle = copy.copy(self.oracle)
      py = np.array(list(self.scaled_oracle.values()))

      self.SCALE_REWARD_MIN = args.scale_reward_min
      self.SCALE_REWARD_MAX = args.scale_reward_max
      self.REWARD_EXP = args.reward_exp

      py = np.maximum(py, self.SCALE_REWARD_MIN)
      py = py ** self.REWARD_EXP
      self.scale = self.SCALE_REWARD_MAX / max(py)
      py = py * self.scale
      
      self.scaled_oracle = {x: y for x, y in zip(self.scaled_oracle.keys(), py)}

      # Rewards
      self.rs_all = [y for x, y in self.scaled_oracle.items()]

      # Modes
      with open('datasets/tfbind8/modes_tfbind8.pkl', 'rb') as f:
        modes = pickle.load(f)
      self.modes = set([self.state(munge(x), is_leaf=True) for x in modes])

      self.reward_dict = {}
    
    def unnormalize(self, r):
      r = r / self.scale
      r = r ** (1 / self.REWARD_EXP)
      return r


    # Core
    def reward(self, x):
      #assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      temp = copy.deepcopy(x)
      if len(x.content) != 8:
        temp = copy.deepcopy(x)
        right_am = '0'*((8-len(x))//2)
        left_am = '0'*((8-len(x))//2)
        if len(x.content) % 2 != 0:
          left_am += '0'
        temp.content = left_am + temp.content + right_am
        temp.is_leaf = True
        return self.scaled_oracle[temp]
      temp.is_leaf = True
      return self.scaled_oracle[temp]

    def is_mode(self, x, r):
      return x in self.modes

    '''
      Interpretation & visualization
    '''
    def dist_func(self, state1, state2):
      """ States are SeqPAState or SeqInsertState objects. """
      return levenshtein(state1.content, state2.content)

    def make_monitor(self):
      target = TargetRewardDistribution()
      target.init_from_base_rewards(self.rs_all)
      return Monitor(self.args, target, dist_func=self.dist_func,
                     is_mode_f=self.is_mode, callback=self.add_monitor, unnormalize = self.unnormalize)


    def add_monitor(self, xs, rs, allXtoR):
      """ Reimplement scoring with oracle, not unscaled oracle (used as R). """
      tolog = dict()
      return tolog

  return TFBind8MDP(args)


def main(args):
  print('Running experiment TFBind8 ...')

  if args.mdp_style == 'pa':
    base = seqpamdp.SeqPrependAppendMDP
    actorclass = seqpamdp.SeqPAActor
  elif args.mdp_style == 'insert':
    base = seqinsertmdp.SeqInsertMDP
    actorclass = seqinsertmdp.SeqInsertActor
  elif args.mdp_style == 'autoregressive':
    base = seqarmdp.SeqAutoregressiveMDP
    actorclass = seqarmdp.SeqARActor
  mdp = dynamic_inherit_mdp(base, args)

  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  # Save memory, after constructing monitor with target rewards
  del mdp.rs_all

  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return

def eval(args):
  print('Running evaluation TFBind8 ...')
  
  if args.mdp_style == 'pa':
    base = seqpamdp.SeqPrependAppendMDP
    actorclass = seqpamdp.SeqPAActor
  elif args.mdp_style == 'insert':
    base = seqinsertmdp.SeqInsertMDP
    actorclass = seqinsertmdp.SeqInsertActor
  elif args.mdp_style == 'autoregressive':
    base = seqarmdp.SeqAutoregressiveMDP
    actorclass = seqarmdp.SeqARActor
  mdp = dynamic_inherit_mdp(base, args)

  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  # Save memory, after constructing monitor with target rewards
  del mdp.rs_all

  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + 'final.pth')
  else:
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + f'round_{args.ckpt}.pth')
    
  # evaluate
  with torch.no_grad():
    eval_samples = model.batch_fwd_sample(args.eval_num_samples, epsilon=0.0)
    
  allXtoR = dict()
  for exp in eval_samples:
    if exp.x not in allXtoR:
      allXtoR[exp.x] = exp.r 
  
  round_num = 1
  monitor.log_samples(round_num, eval_samples)
  log = monitor.eval_samplelog(model, round_num, allXtoR)

  # save results
  result_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    result_path += '/' + 'final.pkl'
  else:
    result_path += '/' + f'round_{args.ckpt}.pkl'
    
  with open(result_path, "wb") as f:
    pickle.dump(log, f)
    
  
