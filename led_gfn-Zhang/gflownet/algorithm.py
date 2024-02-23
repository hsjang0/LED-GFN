import random
import networkx as nx
import copy
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from einops import rearrange, reduce, repeat
import os

from util import get_decided, pad_batch, get_parent
from network import GIN, two_GIN


def sample_from_logits(pf_logits, gb, state, done, rand_prob=0.):
    numnode_per_graph = gb.batch_num_nodes().tolist()
    pf_logits[get_decided(state)] = -np.inf
    pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)

    # use -1 to denote impossible action (e.g. for done graphs)
    action = torch.full([gb.batch_size,], -1, dtype=torch.long, device=gb.device)
    pf_undone = pf_logits[~done].softmax(dim=1)
    action[~done] = torch.multinomial(pf_undone, num_samples=1).squeeze(-1)

    if rand_prob > 0.:
        unif_pf_undone = torch.isfinite(pf_logits[~done]).float()
        rand_action_unodone = torch.multinomial(unif_pf_undone, num_samples=1).squeeze(-1)
        rand_mask = torch.rand_like(rand_action_unodone.float()) < rand_prob
        action[~done][rand_mask] = rand_action_unodone[rand_mask]
    return action


class DetailedBalance(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.task = cfg.task
        self.device = device

        assert cfg.arch in ["gin"]
        gin_dict = {"hidden_dim": cfg.hidden_dim, "num_layers": cfg.hidden_layer,
                    "dropout": cfg.dropout, "learn_eps": cfg.learn_eps,
                    "aggregator_type": cfg.aggr}
        self.model = GIN(3, 1, graph_level_output=0, **gin_dict).to(device)
        self.model_reward = two_GIN(3, 0, graph_level_output=1, **gin_dict).to(device)
        self.model_flow = GIN(3, 0, graph_level_output=1, **gin_dict).to(device)
        self.params = [
            {"params": self.model.parameters(), "lr": cfg.lr},
            {"params": self.model_flow.parameters(), "lr": cfg.zlr},
        ]
        self.optimizer = torch.optim.Adam(self.params)
        self.optimizer_reward = torch.optim.Adam(self.model_reward.parameters(), lr= 1e-3)
        self.leaf_coef = cfg.leaf_coef

    def parameters(self):
        return list(self.model.parameters()) + list(self.model_flow.parameters())

    @torch.no_grad()
    def sample(self, gb, state, done, rand_prob=0., temperature=1., reward_exp=None):
        self.model.eval()
        pf_logits = self.model(gb, state.to(self.device), reward_exp)[..., 0]
        return sample_from_logits(pf_logits / temperature, gb, state, done, rand_prob=rand_prob)

    def save(self, path):
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_dict.update({"model_flow": self.model_flow.state_dict()})
        torch.save(save_dict, path)
        print(f"Saved to {path}")

    def load(self, path):
        save_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(save_dict["model"])
        self.model_flow.load_state_dict(save_dict["model_flow"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        print(f"Loaded from {path}")

    def train_step(self, *batch):
        raise NotImplementedError


class DetailedBalanceTransitionBuffer(DetailedBalance):
    def __init__(self, cfg, device):
        assert cfg.alg in ["db", "fl","led"]
        self.cfg = cfg
        super(DetailedBalanceTransitionBuffer, self).__init__(cfg, device)

    def train_step(self, *batch, reward_exp=None, logr_scaler=None, ep = 1):
        self.model.train()
        self.model_flow.train()
        torch.cuda.empty_cache()

        gb, s, logr, a, s_next, logr_next, d = batch
        gb, s, logr, a, s_next, logr_next, d = gb.to(self.device), s.to(self.device), logr.to(self.device), \
                    a.to(self.device), s_next.to(self.device), logr_next.to(self.device), d.to(self.device)
        
        if self.cfg.alg == 'led':
            logr = self.model_reward(gb, s,s_next).view(-1)

        logr, logr_next = logr_scaler(logr), logr_scaler(logr_next)
        numnode_per_graph = gb.batch_num_nodes().tolist()
        batch_size = gb.batch_size

        total_num_nodes = gb.num_nodes()
        gb_two = dgl.batch([gb, gb])
        s_two = torch.cat([s, s_next], dim=0)
        logits = self.model(gb_two, s_two, reward_exp)
        _, flows_out = self.model_flow(gb_two, s_two, reward_exp) # (2 * num_graphs, 1)
        flows, flows_next = flows_out[:batch_size, 0], flows_out[batch_size:, 0]

        pf_logits = logits[:total_num_nodes, ..., 0]
        pf_logits[get_decided(s)] = -np.inf
        pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)
        log_pf = F.log_softmax(pf_logits, dim=1)[torch.arange(batch_size), a]

        log_pb = torch.tensor([torch.log(1 / get_parent(s_, self.task).sum())
         for s_ in torch.split(s_next, numnode_per_graph, dim=0)]).to(self.device)

        if self.cfg.alg == 'led':
            flows_next.masked_fill_(d, 0.) # \tilde F(x) = F(x) / R(x) = 1, log 1 = 0
            lhs = flows + log_pf 
            rhs = flows_next + log_pb
            loss = (lhs - rhs -logr).pow(2)
            loss = loss.mean()
        elif self.cfg.alg == 'fl':
            flows_next.masked_fill_(d, 0.) # \tilde F(x) = F(x) / R(x) = 1, log 1 = 0
            lhs = logr + flows + log_pf 
            rhs = logr_next + flows_next + log_pb
            loss = (lhs - rhs).pow(2)
            loss = loss.mean()
        else:
            flows_next = torch.where(d, logr_next, flows_next)
            lhs = flows + log_pf # (bs,)
            rhs = flows_next + log_pb
            losses = (lhs - rhs).pow(2)
            loss = (losses[d].sum() * self.leaf_coef + losses[~d].sum()) / batch_size

        return_dict = {"train/loss": loss.item()}
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return return_dict



    def train_step_reward(self, gb, s, reward, reward_exp=None, logr_scaler=None):
        self.model_reward.train()
        torch.cuda.empty_cache()
    
        gb = gb.to(self.device)
        g_list = dgl.unbatch(gb)
        state = s.to(self.device)
        numnode_per_graph = gb.batch_num_nodes().tolist()
        state = torch.split(state, numnode_per_graph, dim=0)
        reward = torch.sum(reward.to(self.device), dim = -1)
        loss = None
        
        for b_idx in range(0,len(state)):
            reward_est = None
            t_mask_r = torch.zeros((1)).to(gb.device)
            est_list = []
            for i in range(1, state[b_idx].shape[1]):
                s, s_next = state[b_idx][:,i-1], state[b_idx][:, i]
                
                if reward_est == None:
                    est = self.model_reward(g_list[b_idx].to(self.device),s,s_next)
                    reward_est = est
                else:
                    est = self.model_reward(g_list[b_idx].to(self.device),s,s_next)
                    reward_est = reward_est + est
                est_list.append(est.item())
            
            if loss == None:
                loss = (reward_est - reward[b_idx]).pow(2).mean()
            else:
                loss = loss+ (reward_est - reward[b_idx]).pow(2).mean()
            
            if (b_idx+1) % 4 == 0:
                self.optimizer_reward.zero_grad()
                return_dict = {"train/loss": loss.item()}    
                (loss/4).backward()
                self.optimizer_reward.step()
                loss = None
        return return_dict
    





