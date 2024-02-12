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

import model_atom, model_block, model_fingerprint
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from data import Dataset, DatasetDirect, get_mol_path_graph, make_model

import copy
from tqdm import tqdm
import datetime
import sys, os





class Proxy:
    def __init__(self, args, bpath, device):
        eargs = pickle.load(gzip.open(f'{args.proxy_path}/info.pkl.gz'))['args']
        params = pickle.load(gzip.open(f'{args.proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, eargs.repr_type)
        self.mdp.floatX = args.floatX
        self.proxy = make_model(eargs, self.mdp)
        print ('proxy', self.proxy)

        # If you get an error when loading the proxy parameters, it is probably due to a version
        # mismatch in torch geometric. Try uncommenting this code instead of using the super_hackish_param_map

        for a, b in zip(self.proxy.parameters(), params):
           a.data = torch.tensor(b, dtype=self.mdp.floatX)

        super_hackish_param_map = {
            'mpnn.lin0.weight': params[0],
            'mpnn.lin0.bias': params[1],
            'mpnn.conv.bias': params[3],
            'mpnn.conv.nn.0.weight': params[4],
            'mpnn.conv.nn.0.bias': params[5],
            'mpnn.conv.nn.2.weight': params[6],
            'mpnn.conv.nn.2.bias': params[7],
            'mpnn.conv.lin.weight': params[2],
            'mpnn.gru.weight_ih_l0': params[8],
            'mpnn.gru.weight_hh_l0': params[9],
            'mpnn.gru.bias_ih_l0': params[10],
            'mpnn.gru.bias_hh_l0': params[11],
            'mpnn.lin1.weight': params[12],
            'mpnn.lin1.bias': params[13],
            'mpnn.lin2.weight': params[14],
            'mpnn.lin2.bias': params[15],
            'mpnn.set2set.lstm.weight_ih_l0': params[16],
            'mpnn.set2set.lstm.weight_hh_l0': params[17],
            'mpnn.set2set.lstm.bias_ih_l0': params[18],
            'mpnn.set2set.lstm.bias_hh_l0': params[19],
            'mpnn.lin3.weight': params[20],
            'mpnn.lin3.bias': params[21],
        }
        # for k, v in super_hackish_param_map.items():
        #     self.proxy.get_parameter(k).data = torch.tensor(v, dtype=self.mdp.floatX)

        self.proxy.to(device)

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        t0 = time.time()
        proxy_out = self.proxy(m, do_stems=False)[1].item()
        t1 = time.time()
        # print ('eval m time: {}s'.format(t1 - t0))
        return proxy_out

_stop = [None]


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