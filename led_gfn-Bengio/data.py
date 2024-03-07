import argparse
import gzip
import os
import pdb
import pickle
import threading
import time
import warnings
from copy import deepcopy

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import heapq
from rdkit import DataStructs

import model_atom, model_block, model_fingerprint
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended

import copy
from tqdm import tqdm
import datetime
import sys, os
from utils import chem

from rdkit import Chem

class Dataset:
    def __init__(self, args, bpath, device, floatX=torch.double):
        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(int(time.time()))
        self.train_mols = []
        self.test_mols = []
        self.train_mols_map = {}
        self.is_visit = {}
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.repr_type, include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        self._device = device
        self.seen_molecules = set()
        self.stop_event = threading.Event()
        self.target_norm = [-8.6, 1.10]
        self.sampling_model = None
        self.sampling_model_prob = 0
        self.floatX = floatX
        self.mdp.floatX = self.floatX
        self.good = 0
        
        
        #######
        # This is the "result", here a list of (reward, BlockMolDataExt, info...) tuples
        self.sampled_mols = []

        get = lambda x, d: getattr(args, x) if hasattr(args, x) else d
        self.min_blocks = get('min_blocks', 2)
        self.max_blocks = get('max_blocks', 10)
        self.mdp._cue_max_blocks = self.max_blocks
        self.replay_mode = get('replay_mode', 'dataset')
        self.reward_exp = get('reward_exp', 1)
        self.reward_norm = get('reward_norm', 1)
        self.random_action_prob = get('random_action_prob', 0)
        self.R_min = get('R_min', 1e-8)
        self.ignore_parents = get('ignore_parents', False)
        self.early_stop_reg = get('early_stop_reg', 0)

        self.top_100 = []
        self.num_calls = 0
        self.num_calls_act = 0
        self.online_mols = []
        self.max_online_mols = 30000

        self.sparse_r = args.use_sp_r if hasattr(args, 'use_sp_r') else 0
        self.sparse_r_threshold = args.sp_r_thres if hasattr(args, 'sp_r_thres') else -1.

        self.fl = args.fl if hasattr(args, 'fl') else 0

    def _get(self, i, dset):
        if ((self.sampling_model_prob > 0 and self.train_rng.uniform() < self.sampling_model_prob) or len(dset) < 32):
            # don't sample if we don't have to
            return self._get_sample_model()

        # Sample trajectories by walking backwards from the molecules in our dataset
        # Handle possible multithreading issues when independent threads add/substract from dset:
        while True:
            try:
                m = dset[i]
            except IndexError:
                i = self.train_rng.randint(0, len(dset))
                continue
            break

        if not isinstance(m, BlockMoleculeDataExtended):
            m = m[-1]

        r = m.reward
        done = 1

        samples = []
        # a sample is a tuple (parents(s), parent actions, reward(s), s, done)
        # an action is (blockidx, stemidx) or (-1, x) for 'stop'
        # so we start with the stop action, unless the molecule is already a "terminal" node (if it has no stems, no actions).
        if len(m.stems):
            samples.append(((m,), ((-1, 0),), r, m, done))
            r = done = 0
        while len(m.blocks): # and go backwards
            parents, actions = zip(*self.mdp.parents(m))
            samples.append((parents, actions, r, m, done))
            r = done = 0
            m = parents[self.train_rng.randint(len(parents))]

        return samples

    def set_sampling_model(self, model, proxy_reward, sample_prob=0.5):
        self.sampling_model = model
        self.sampling_model_prob = sample_prob
        self.proxy_reward = proxy_reward

    def _get_sample_model(self):
        m = BlockMoleculeDataExtended()

        samples = []
        
        max_blocks = self.max_blocks
        
        if self.early_stop_reg > 0 and np.random.uniform() < self.early_stop_reg:
            early_stop_at = np.random.randint(self.min_blocks, self.max_blocks + 1)
        else:
            early_stop_at = max_blocks + 1
        
        trajectory_stats = []
        for t in range(max_blocks):
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])

            s_o, m_o = self.sampling_model(s)
            
            ## fix from run 330 onwards
            if t < self.min_blocks:
                m_o = m_o * 0 - 1000 # prevent assigning prob to stop when we can't stop
            ##
            
            logits = torch.cat([m_o[:, 0].reshape(-1), s_o.reshape(-1)])
            
            #print(m_o.shape, s_o.shape, logits.shape)
            #print(m.blockidxs, m.jbonds, m.stems)
            
            cat = torch.distributions.Categorical(logits=logits)
            action = cat.sample().item()
            #print(action)

            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(t < self.min_blocks), logits.shape[0])

            if t == early_stop_at:
                action = 0

            q = torch.cat([m_o[:, 0].reshape(-1), s_o.reshape(-1)])

            trajectory_stats.append((q[action].item(), action, torch.logsumexp(q, 0).item()))
            
            if t >= self.min_blocks and action == 0:
                r = self._get_reward(m, save_true=True)
                if self.fl:
                    r_fl = self._get_reward(m, save_true=True)
                    samples.append(((m,), ((-1, 0),), r, r_fl, None, 1))
                else:
                    samples.append(((m,), ((-1, 0),), r, None, 1))
                break
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                #print('..', action)
                m_old = m
                m = self.mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r = self._get_reward(m, save_true=True)
                    if self.fl:
                        r_fl = r
                        if self.ignore_parents:
                            samples.append(((m_old,), (action,), r, r_fl, m, 1))
                        else:
                            samples.append((*zip(*self.mdp.parents(m)), r, r_fl, m, 1))
                    else:
                        if self.ignore_parents:
                            samples.append(((m_old,), (action,), r, m, 1))
                        else:
                            samples.append((*zip(*self.mdp.parents(m)), r, m, 1))
                    break
                else:
                    if self.fl:
                        r_fl = self._get_reward(m)
                        if self.ignore_parents:
                            samples.append(((m_old,), (action,), 0, r_fl, m, 0))
                        else:
                            samples.append((*zip(*self.mdp.parents(m)), 0, r_fl, m, 0))
                    else:
                        if self.ignore_parents:
                            samples.append(((m_old,), (action,), 0, m, 0))
                        else:
                            samples.append((*zip(*self.mdp.parents(m)), 0, m, 0))
        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in samples[-1][0]])
        qp = self.sampling_model(p, None)
        qsa_p = self.sampling_model.index_output_by_action(
            p, qp[0], qp[1][:, 0], torch.tensor(samples[-1][1], device=self._device).long()
        )
        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        self.sampled_mols.append((r, m, trajectory_stats, inflow))
        if self.replay_mode == 'online' or self.replay_mode == 'prioritized':
            m.reward = r
            self._add_mol_to_online(r, m, inflow)
        return samples

    def _add_mol_to_online(self, r, m, inflow):
        if self.replay_mode == 'online':
            r = r + self.train_rng.normal() * 0.01
            if len(self.online_mols) < self.max_online_mols or r > self.online_mols[0][0]:
                self.online_mols.append((r, m))
            if len(self.online_mols) > self.max_online_mols:
                self.online_mols = sorted(self.online_mols)[max(int(0.05 * self.max_online_mols), 1):]
                #print(r)
        elif self.replay_mode == 'prioritized':
            self.online_mols.append((abs(inflow - np.log(r)), m))
            if len(self.online_mols) > self.max_online_mols * 1.1:
                self.online_mols = self.online_mols[-self.max_online_mols:]


    def _get_reward(self, m, save_true=False):
        rdmol = m.mol
        if rdmol is None:
            return self.R_min

        self.num_calls += 1
        smi = m.smiles
            
        if smi in self.train_mols_map:
            
            a = self.train_mols_map[smi]
            if save_true and not (smi in self.is_visit):
                heapq.heappush(self.top_100, (a,m))
                if a >= 7.5:
                    self.good += 1
                    
                self.is_visit[smi] = True
                if len(self.top_100) > 100:
                    heapq.heappop(self.top_100)

            return self.r2r(normscore=a)
        
        
        a = self.proxy_reward(m)
        self.num_calls_act += 1 
        self.train_mols_map[smi] = a 
        
        if save_true and not (smi in self.is_visit):
            heapq.heappush(self.top_100, (a,m))
            if a >= 7.5:
                self.good += 1
            
            self.is_visit[smi] = True
            if len(self.top_100) > 100:
                heapq.heappop(self.top_100)
                
        return self.r2r(normscore=a)

    def top_100_rwd(self):
        top_scores = [score for score, _ in self.top_100]
        return np.sum(top_scores)/100

    def tasimoto(self):
        dists =[]
        for e, (_,m1) in enumerate(self.top_100):
            for (_,m2) in (self.top_100[e+1:]):
                dist = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m1.mol), Chem.RDKFingerprint(m2.mol))
                dists.append(dist)
        return np.mean(dists)

    def sample(self, n):
        if self.replay_mode == 'dataset':
            eidx = self.train_rng.randint(0, len(self.train_mols), n)
            samples = sum((self._get(i, self.train_mols) for i in eidx), [])
        elif self.replay_mode == 'online':
            eidx = self.train_rng.randint(0, max(1, len(self.online_mols)), n)

            # calls the "_get_sample_model" function
            samples = sum((self._get(i, self.online_mols) for i in eidx), [])
        elif self.replay_mode == 'prioritized':
            if not len(self.online_mols):
                # _get will sample from the model
                samples = sum((self._get(0, self.online_mols) for i in range(n)), [])
            else:
                prio = np.float32([i[0] for i in self.online_mols])
                eidx = self.train_rng.choice(len(self.online_mols), n, False, prio/prio.sum())
                samples = sum((self._get(i, self.online_mols) for i in eidx), [])
        return zip(*samples)

    def sample2batch(self, mb):
        if self.fl:
            p, a, r, r_fl, s, d, *o = mb
        else:
            p, a, r, s, d, *o = mb

        mols = (p, s)

        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i,p in enumerate(p)], []), device=self._device).long()
        
        # Convert all parents and states to repr. Note that this concatenates all the parent lists, which is why we need p_batch
        p = self.mdp.mols2batch(list(map(self.mdp.mol2repr, sum(p, ()))))
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        
        # Concatenate all the actions (one per parent per sample)
        a = torch.tensor(sum(a, ()), device=self._device).long()
        
        # rewards and dones
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        
        if self.fl:
            r_fl = torch.tensor(r_fl, device=self._device).to(self.floatX)
            return (p, p_batch, a, r, r_fl, s, d, mols, *o)
        else:
            return (p, p_batch, a, r, s, d, mols, *o)

    def r2r(self, dockscore=None, normscore=None):
        #print(normscore)
        if dockscore is not None:
            normscore = 4 - (min(0, dockscore) - self.target_norm[0]) / self.target_norm[1]
        
        normscore = max(self.R_min, normscore)
        transformed_r = (normscore / self.reward_norm) ** self.reward_exp

        return transformed_r

    def start_samplers(self, n, mbsize):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n
        
        def f(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(self.sample(mbsize))
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()

        self.sampler_threads = [threading.Thread(target=f, args=(i,)) for i in range(n)]
        [setattr(i, 'failed', False) for i in self.sampler_threads]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
          while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]

class DatasetDirect(Dataset):
    def sample(self, n):
        trajectories = [self._get_sample_model() for i in range(n)]
        batch = (*zip(*sum(trajectories, [])), sum([[i] * len(t) for i, t in enumerate(trajectories)], []), [len(t) for t in trajectories])
        return batch

    def sample2batch(self, mb):
        if self.fl:
            s, a, r, r_fl, sp, d, idc, lens = mb
        else:
            s, a, r, sp, d, idc, lens = mb

        mols = (s, sp)
        s = self.mdp.mols2batch([self.mdp.mol2repr(i[0]) for i in s])
        a = torch.tensor(sum(a, ()), device=self._device).long()
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        n = torch.tensor([len(self.mdp.parents(m)) if (m is not None) else 1 for m in sp], device=self._device).to(self.floatX)
        idc = torch.tensor(idc, device=self._device).long()
        lens = torch.tensor(lens, device=self._device).long()

        if self.fl:
            r_fl = torch.tensor(r_fl, device=self._device).to(self.floatX)
            return (s, a, r, r_fl, d, n, mols, idc, lens)
        else:
            return (s, a, r, d, n, mols, idc, lens)

def make_model(args, mdp, out_per_mol=1):
    if args.repr_type == 'block_graph':
        model = model_block.GraphAgent(
            nemb=args.nemb,
            nvec=0,
            out_per_stem=mdp.num_blocks,
            out_per_mol=out_per_mol,
            num_conv_steps=args.num_conv_steps,
            mdp_cfg=mdp,
            version=args.model_version,
        )
    elif args.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(
            nhid=args.nemb,
            nvec=0,
            num_out_per_stem=mdp.num_blocks,
            num_out_per_mol=out_per_mol,
            num_conv_steps=args.num_conv_steps,
            version=args.model_version,
            do_nblocks=(hasattr(args,'include_nblocks') and args.include_nblocks), 
            dropout_rate=0.1
        )
    elif args.repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
    
    return model



def get_mol_path_graph(mol):
    bpath = "data/blocks_PDB_105.json"

    mdp = MolMDPExtended(bpath)
    mdp.post_init(torch.device('cpu'), 'block_graph')
    mdp.build_translation_table()
    mdp.floatX = torch.float
    
    agraph = nx.DiGraph()
    agraph.add_node(0)
    
    ancestors = [mol]
    ancestor_graphs = []

    par = mdp.parents(mol)
    mstack = [i[0] for i in par]
    pstack = [[0, a] for i,a in par]
    while len(mstack):
        m = mstack.pop() # pop = last item is default index
        p, pa = pstack.pop()
        match = False
        mgraph = mdp.get_nx_graph(m)
        for ai, a in enumerate(ancestor_graphs):
            if mdp.graphs_are_isomorphic(mgraph, a):
                agraph.add_edge(p, ai+1, action=pa)
                match = True
                break
        if not match:
            agraph.add_edge(p, len(ancestors), action=pa) # I assume the original molecule = 0, 1st ancestor = 1st parent = 1
            ancestors.append(m) # so now len(ancestors) will be 2 --> and the next edge will be to the ancestor labelled 2
            ancestor_graphs.append(mgraph)
            if len(m.blocks):
                par = mdp.parents(m)
                mstack += [i[0] for i in par]
                pstack += [(len(ancestors)-1, i[1]) for i in par]

    for u, v in agraph.edges:
        c = mdp.add_block_to(ancestors[v], *agraph.edges[(u,v)]['action'])
        geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True), mdp.get_nx_graph(ancestors[u], true_block=True))
        if not geq: # try to fix the action
            block, stem = agraph.edges[(u,v)]['action']
            for i in range(len(ancestors[v].stems)):
                c = mdp.add_block_to(ancestors[v], block, i)
                geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True), mdp.get_nx_graph(ancestors[u], true_block=True))
                if geq:
                    agraph.edges[(u,v)]['action'] = (block, i)
                    break
        if not geq:
            raise ValueError('could not fix action')
    for u in agraph.nodes:
        agraph.nodes[u]['mol'] = ancestors[u]
    return agraph
    
try:
    from arrays import*
except:
    print("no arrays")
