import sys, os
import gzip, pickle
from time import time, sleep
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf


import random
import numpy as np
import torch
import dgl
from einops import rearrange, reduce, repeat

from data import get_data_loaders
from util import seed_torch, TransitionBuffer, get_mdp_class
from algorithm import DetailedBalanceTransitionBuffer
import wandb

#torch.backends.cudnn.benchmark = True

def get_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl", "led"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, cfg.device)
    return alg, buffer

def get_logr_scaler(cfg, process_ratio=1., reward_exp=None):
    if reward_exp is None:
        reward_exp = float(cfg.reward_exp)

    if cfg.anneal == "linear":
        process_ratio = max(0., min(1., process_ratio)) # from 0 to 1
        reward_exp = reward_exp * process_ratio +\
                     float(cfg.reward_exp_init) * (1 - process_ratio)
    elif cfg.anneal == "none":
        pass
    else:
        raise NotImplementedError

    # (R/T)^beta -> (log R - log T) * beta
    def logr_scaler(sol_size, gbatch=None):
        logr = sol_size
        return logr * reward_exp
    return logr_scaler

def refine_cfg(cfg):
    with open_dict(cfg):
        cfg.device = cfg.d
        cfg.work_directory = os.getcwd()

        if cfg.task in ["mis", "maxindset", "maxindependentset",]:
            cfg.task = "MaxIndependentSet"
            cfg.wandb_project_name = "MIS"
        elif cfg.task in ["mds", "mindomset", "mindominateset",]:
            cfg.task = "MinDominateSet"
            cfg.wandb_project_name = "MDS"
        elif cfg.task in ["mc", "maxclique",]:
            cfg.task = "MaxClique"
            cfg.wandb_project_name = "MaxClique"
        elif cfg.task in ["mcut", "maxcut",]:
            cfg.task = "MaxCut"
            cfg.wandb_project_name = "MaxCut"
        else:
            raise NotImplementedError

        # architecture
        assert cfg.arch in ["gin"]

        # log reward shape
        cfg.reward_exp = cfg.rexp
        cfg.reward_exp_init = cfg.rexpit
        if cfg.anneal in ["lin"]:
            cfg.anneal = "linear"

        # training
        cfg.batch_size = cfg.bs
        cfg.batch_size_interact = cfg.bsit
        cfg.leaf_coef = cfg.lc
        cfg.same_graph_across_batch = cfg.sameg

        # data
        cfg.test_batch_size = cfg.tbs
        if "rb" in cfg.input:
            cfg.data_type = cfg.input.upper()
        elif "ba" in cfg.input:
            cfg.data_type = cfg.input.upper()
        else:
            raise NotImplementedError

    del cfg.d, cfg.rexp, cfg.rexpit, cfg.bs, cfg.bsit, cfg.lc, cfg.sameg, cfg.tbs
    return cfg

@torch.no_grad()
def rollout(gbatch, cfg, alg, reward_est = None):
    env = get_mdp_class(cfg.task)(gbatch, cfg)
    state = env.state

    if not (reward_est is None):
        reward_est.eval()
    ##### sample traj
    reward_exp_eval = None
    traj_s, traj_r, traj_a, traj_d = [], [], [], []
    while not all(env.done):
        action = alg.sample(gbatch, state, env.done, rand_prob=cfg.randp, reward_exp=reward_exp_eval)

        traj_a.append(action)
        traj_d.append(env.done)
        traj_r.append(env.get_log_reward())
        traj_s.append(state)
        state = env.step(action)

    traj_r.append(env.get_log_reward())
    traj_s.append(state)
    traj_d.append(env.done)

    traj_s = torch.stack(traj_s, dim=1) # (sum of #node per graph in batch, max_traj_len)
    traj_r = torch.stack(traj_r, dim=1) # (batch_size, max_traj_len)
    traj_a = torch.stack(traj_a, dim=1) # (batch_size, max_traj_len-1)
    """
    traj_a is tensor like 
    [ 4, 30, 86, 95, 96, 29, -1, -1],
    [47, 60, 41, 11, 55, 64, 80, -1],
    [26, 38, 13,  5,  9, -1, -1, -1]
    """
    traj_d = torch.stack(traj_d, dim=1) # (batch_size, max_traj_len)
    """
    traj_d is tensor like 
    [False, False, False, False, False, False,  True,  True,  True],
    [False, False, False, False, False, False, False,  True,  True],
    [False, False, False, False, False,  True,  True,  True,  True]
    """
    traj_len = 1 + torch.sum(~traj_d, dim=1) # (batch_size, )

    ##### graph, state, action, done, reward, trajectory length
    batch = gbatch.cpu(), traj_s.cpu(), traj_a.cpu(), traj_d.cpu(), traj_r.cpu(), traj_len.cpu()
    return batch, env.batch_metric(state)


@hydra.main(config_path="configs", config_name="main") # for hydra-core==1.1.0
# @hydra.main(version_base=None, config_path="configs", config_name="main") # for newer hydra
def main(cfg: DictConfig):
    cfg = refine_cfg(cfg)
    
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.wandb_project_name,
        name=cfg.alg+'_'+str(cfg.input),
        # track hyperparameters and run metadata
    )
    

    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    alg, buffer = get_alg_buffer(cfg, device)
    seed_torch(cfg.seed)
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    train_loader, test_loader = get_data_loaders(cfg)


    trainset_size = len(train_loader.dataset)
    print(f"Trainset size: {trainset_size}")
    alg_save_path = os.path.abspath("./alg.pt")
    alg_save_path_best = os.path.abspath("./alg_best.pt")
    train_data_used = 0
    train_step = 0
    train_logr_scaled_ls = []
    train_metric_ls = []
    metric_best = 0.
    result = {"set_size": {}, "logr_scaled": {}, "train_data_used": {}, "train_step": {}, }

    @torch.no_grad()
    def evaluate(ep, train_step, train_data_used, logr_scaler):
        torch.cuda.empty_cache()
        num_repeat = 20 # 50
        mis_ls, mis_top1_ls, num_modes, mis_top10_ls = [], [], [], []
        logr_ls = []
        pbar = tqdm(enumerate(test_loader))
        pbar.set_description(f"Test Epoch {ep:2d} Data used {train_data_used:5d}")
        for batch_idx, gbatch in enumerate(test_loader):
            gbatch = gbatch.to(device)
            gbatch_rep = dgl.batch([gbatch] * num_repeat)

            env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
            state = env.state
            while not all(env.done):
                action = alg.sample(gbatch_rep, state, env.done, rand_prob=0.)
                state = env.step(action)

            logr_rep = logr_scaler(env.get_log_reward())
            logr_ls += logr_rep.tolist()
            
            batch_state = torch.split(state, env.numnode_per_graph, dim=0)
            max_length = max(len(vector) for vector in batch_state)
            padded_state = torch.zeros((len(batch_state), max_length)).to(batch_state[0].device)
            for i, vector in enumerate(batch_state):
                padded_state[i, :len(vector)] += vector
            batch_state = padded_state
            
            curr_mis_rep = torch.tensor(env.batch_metric(state))
            curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
            mis_ls += curr_mis_rep.mean(dim=1).tolist()

        print(f"Test Epoch{ep:2d} Data used{train_data_used:5d}: "
              f"Metric={np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}, "
              f"LogR scaled={np.mean(logr_ls):.2e}+-{np.std(logr_ls):.2e}")

        wandb.log({
            "metric size_base": np.mean(mis_ls)
        })

        result["set_size"][ep] = np.mean(mis_ls)
        result["logr_scaled"][ep] = np.mean(logr_ls)
        result["train_step"][ep] = train_step
        result["train_data_used"][ep] = train_data_used
        pickle.dump(result, gzip.open("./result.json", 'wb'))


    for ep in range(cfg.epochs):
        temporal = []
        for batch_idx, gbatch in enumerate(train_loader):
            reward_exp = None
            process_ratio = max(0., min(1., train_data_used / cfg.annend))
            logr_scaler = get_logr_scaler(cfg, process_ratio=process_ratio, reward_exp=reward_exp)

            train_logr_scaled_ls = train_logr_scaled_ls[-5000:]
            train_metric_ls = train_metric_ls[-5000:]
            
            gbatch = gbatch.to(device)
            train_data_used += gbatch.batch_size
            
            ###### rollout
            if (cfg.alg == 'led'):
                batch_, metric_ls = rollout(gbatch, cfg, alg)
                temporal.append(batch_)
            else:
                batch_, metric_ls = rollout(gbatch, cfg, alg)
            
            buffer.add_batch(batch_)

            ##### train
            batch_size = min(len(buffer), cfg.batch_size)
            indices = list(range(len(buffer)))
            
            if (cfg.alg == 'led'): #Learning energy decomposition
                gbatch, traj_s, traj_a, traj_d, traj_r, traj_len = temporal[random.randint(0,len(temporal)-1)]
                train_info = alg.train_step_reward(gbatch, traj_s, traj_r, reward_exp=reward_exp, logr_scaler=logr_scaler)

            
            for _ in range(cfg.tstep):
                if len(indices) == 0:
                    break
                curr_indices = random.sample(indices, min(len(indices), batch_size))
                batch = buffer.sample_from_indices(curr_indices)
                train_info = alg.train_step(*batch, reward_exp=reward_exp, 
                                            logr_scaler=logr_scaler, ep=ep)
                indices = [i for i in indices if i not in curr_indices]

            if cfg.onpolicy:
                buffer.reset()

            ##### eval
            if batch_idx == 0 or train_step % cfg.eval_freq == 0:
                alg.save(alg_save_path)
                metric_curr = np.mean(train_metric_ls[-1000:])
                if metric_curr > metric_best:
                    metric_best = metric_curr
                    alg.save(alg_save_path_best)
                if cfg.eval:
                    evaluate(ep, train_step, train_data_used, logr_scaler)

            train_step += 1

    evaluate(cfg.epochs, train_step, train_data_used, logr_scaler)
    alg.save(alg_save_path)


if __name__ == "__main__":
    main()