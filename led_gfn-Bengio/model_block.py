import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

import copy

class StateEmbeddingNet(nn.Module):
    def __init__(self, nemb, nvec, num_conv_steps, mdp_cfg, version='v1'):
        super().__init__()

        if version == 'v5': version = 'v4'
        self.version = version

        self.embeddings = nn.ModuleList([
            nn.Embedding(mdp_cfg.num_true_blocks + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types, nemb)
        ])

        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean')
        
        nvec_1 = nvec * (version == 'v1' or version == 'v3')
        nvec_2 = nvec * (version == 'v2' or version == 'v3')
        
        self.block2emb = nn.Sequential(nn.Linear(nemb + nvec_1, nemb), nn.LeakyReLU(), nn.Linear(nemb, nemb))
        
        self.gru = nn.GRU(nemb, nemb)

        self.num_conv_steps = num_conv_steps
        self.nemb = nemb

    def forward(self, graph_data, vec_data=None):
        blockemb, _, bondemb = self.embeddings

        graph_data.x = blockemb(graph_data.x)
        
        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2))

        out = graph_data.x

        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        else:  # if self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out)

        h = out.unsqueeze(0)
        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # print ('out', out.shape, 'graph_data.batch', graph_data.batch.shape)
        # print ('graph_data', graph_data.batch)

        global_mean_pool_out = gnn.global_mean_pool(out, graph_data.batch)
        # mol_preds = self.global2pred(global_mean_pool_out)

        state_embedding = global_mean_pool_out

        return state_embedding

class RND(nn.Module):
    def __init__(self, nemb, nvec, num_conv_steps, mdp_cfg, version='v1', ri_coe=1.0):
        super().__init__()

        self.random_target_network = StateEmbeddingNet(nemb, nvec, num_conv_steps, mdp_cfg, version)
        self.predictor_network = StateEmbeddingNet(nemb, nvec, num_conv_steps, mdp_cfg, version)

        self.reward_scale = ri_coe
    
    def forward(self, graph_data):
        graph_data_copy = copy.deepcopy(graph_data) # added

        random_s_emb = self.random_target_network(graph_data)

        predicted_s_emb = self.predictor_network(graph_data_copy)

        return random_s_emb, predicted_s_emb

    def compute_intrinsic_reward(self, graph_data):
        random_s_emb, predicted_s_emb = self.forward(graph_data)

        intrinsic_reward = torch.norm(predicted_s_emb.detach() - random_s_emb.detach(), dim=-1, p=2)
        intrinsic_reward *= self.reward_scale

        # intrinsic_reward = intrinsic_reward #.cpu().detach().numpy()

        return intrinsic_reward

    def compute_loss(self, graph_data):
        random_s_emb, predicted_s_emb = self.forward(graph_data)
        rnd_loss = torch.norm(predicted_s_emb - random_s_emb.detach(), dim=-1, p=2)
        mean_rnd_loss = torch.mean(rnd_loss)
        return mean_rnd_loss

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, s_latent_dim=128, a_latent_dim=128, reward_scale=0.01, fw_coe=0.2, enc_a='emb'):
        super(ICM, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )
        print ('state_encoder', self.encoder)

        self.enc_a = enc_a
        if self.enc_a != 'ohe':
            self.action_encoder = nn.Embedding(action_dim, a_latent_dim)
            print ('action_encoder', self.action_encoder)
            a_rep_dim = a_latent_dim
        else:
            a_rep_dim = action_dim

        self.forward_model = nn.Sequential(
            nn.Linear(a_rep_dim + s_latent_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim)
        )
        print ('forward_model', self.forward_model)

        self.inverse_model = nn.Sequential(
            nn.Linear(s_latent_dim * 2, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        print ('inverse_model', self.inverse_model)
        
        self.reward_scale = reward_scale
        self.fw_coe = fw_coe
        self.action_dim = action_dim

        self.inv_loss = nn.CrossEntropyLoss()

    def forward(self, state, next_state, action):
        '''
            state: batch_size, state_dim
            next_state: batch_size, state_dim
            action: batch_size
        '''
        # action = torch.nn.functional.one_hot(action, self.action_dim).view(action.shape[0], -1).float()

        phi_s = self.encoder(state) # batch_size, s_latent_dim
        phi_s_next = self.encoder(next_state) # batch_size, s_latent_dim

        if self.enc_a == 'ohe':
            a_rep = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            a_rep = self.action_encoder(action.long()) # batch_size, a_latent_dim

        encoded_sa = torch.cat((a_rep, phi_s), dim=-1) # batch_size, s_latent_dim + a_latent_dim
        hat_phi_s_next = self.forward_model(encoded_sa) # batch_size, s_latent_dim

        cat_s = torch.cat((phi_s, phi_s_next), dim=-1) # batch_size, 2 x s_latent_dim
        hat_a = self.inverse_model(cat_s) # batch_size, action_dim

        return phi_s_next, hat_phi_s_next, hat_a

    def compute_intrinsic_reward(self, states, next_states, actions):
        next_states_latent, next_states_latent_pred, _ = self.forward(states, next_states, actions)
        
        intrinsic_reward = self.reward_scale / 2 * (next_states_latent_pred - next_states_latent).norm(2, dim=-1).pow(2)
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()

        return intrinsic_reward

    def compute_loss(self, states, next_states, actions):
        next_states_latent, next_states_latent_pred, actions_pred = self.forward(states, next_states, actions)
        
        forward_loss = 0.5 * (next_states_latent_pred - next_states_latent.detach()).norm(2, dim=-1).pow(2).mean()
        inverse_loss = self.inv_loss(actions_pred, actions.long())

        curiosity_loss = self.fw_coe * forward_loss + (1 - self.fw_coe) * inverse_loss
        return curiosity_loss, forward_loss, inverse_loss

class GraphAgent(nn.Module):
    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, version='v1'):
        super().__init__()

        if version == 'v5': version = 'v4'
        self.version = version

        self.embeddings = nn.ModuleList([
            nn.Embedding(mdp_cfg.num_true_blocks + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types, nemb)
        ])

        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean')
        
        nvec_1 = nvec * (version == 'v1' or version == 'v3')
        nvec_2 = nvec * (version == 'v2' or version == 'v3')
        
        self.block2emb = nn.Sequential(nn.Linear(nemb + nvec_1, nemb), nn.LeakyReLU(), nn.Linear(nemb, nemb))
        
        self.gru = nn.GRU(nemb, nemb)

        self.stem2pred = nn.Sequential(
            nn.Linear(nemb * 2 + nvec_2, nemb), 
            nn.LeakyReLU(), 
            nn.Linear(nemb, nemb), 
            nn.LeakyReLU(), 
            nn.Linear(nemb, out_per_stem)
        )

        self.global2pred = nn.Sequential(
            nn.Linear(nemb, nemb), 
            nn.LeakyReLU(), 
            nn.Linear(nemb, out_per_mol)
        )
        
        # self.set2set = Set2Set(nemb, processing_steps=3)
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb
        self.training_steps = 0
        self.categorical_style = 'softmax'
        self.escort_p = 6

    def forward(self, graph_data, vec_data=None, do_stems=True):
        blockemb, stememb, bondemb = self.embeddings

        # print ('GraphAgent', graph_data.x)

        graph_data.x = blockemb(graph_data.x)
        
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes)

        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2))

        out = graph_data.x

        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        else:  # if self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out)

        h = out.unsqueeze(0)
        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # Index of the origin block of each stem in the batch (each stem is a pair [block idx, stem atom type], we need to adjust for the batch packing)
        if do_stems:
            if hasattr(graph_data, '_slice_dict'):
                x_slices = torch.tensor(graph_data._slice_dict['x'], device=out.device)[graph_data.stems_batch]
            else:
                x_slices = torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
            
            stem_block_batch_idx = (x_slices + graph_data.stems[:, 0])
            
            if self.version == 'v1' or self.version == 'v4':
                stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)
            elif self.version == 'v2' or self.version == 'v3':
                stem_out_cat = torch.cat(
                    [out[stem_block_batch_idx], graph_data.stemtypes, vec_data[graph_data.stems_batch]],
                    1
                )

            stem_preds = self.stem2pred(stem_out_cat)
        else:
            stem_preds = None

        global_mean_pool_out = gnn.global_mean_pool(out, graph_data.batch)
        mol_preds = self.global2pred(global_mean_pool_out)

        return stem_preds, mol_preds

    def out_to_policy(self, s, stem_o, mol_o):
        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o)
            mol_e = torch.exp(mol_o[:, 0])
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            mol_e = abs(mol_o[:, 0])**self.escort_p
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
        mol_p, stem_p = self.out_to_policy(s, stem_o, mol_o)

        # log-softmax
        mol_lsm = torch.log(mol_p + 1e-20)
        stem_lsm = torch.log(stem_p + 1e-20)

        return -self.index_output_by_action(s, stem_lsm, mol_lsm, a)

    def index_output_by_action(self, s, stem_o, mol_o, a):
        if hasattr(s, '_slice_dict'):
            stem_slices = torch.tensor(s._slice_dict['stems'][:-1], dtype=torch.long, device=stem_o.device)
        else:
            stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
        
        return (stem_o[stem_slices + a[:, 1]][torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0) + mol_o * (a[:, 0] == -1))

    def sum_output(self, s, stem_o, mol_o):
        # s.stems_batch: N (assigns each node to a specific example, with a total of K examples)
        # stem_o: N  |A|
        # mol_o: K
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o

def mol2graph(mol, mdp, floatX=torch.float, bonds=False, nblocks=False):
    f = lambda x: torch.tensor(x, dtype=torch.long, device=mdp.device)
    
    if len(mol.blockidxs) == 0:
        data = Data(  # There's an extra block embedding for the empty molecule
            x=f([mdp.num_true_blocks]),
            edge_index=f([[], []]),
            edge_attr=f([]).reshape((0, 2)),
            stems=f([(0, 0)]),
            stemtypes=f([mdp.num_stem_types]))  # also extra stem type embedding
        return data

    edges = [(i[0], i[1]) for i in mol.jbonds]
    # edge_attrs = [mdp.bond_type_offset[i[2]] +  i[3] for i in mol.jbonds]
    
    t = mdp.true_blockidx
    
    if 0:
        edge_attrs = [((mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[2]) * mdp.num_stem_types +
                       (mdp.stem_type_offset[t[mol.blockidxs[i[1]]]] + i[3]))
                      for i in mol.jbonds]
    else:
        edge_attrs = [(mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[2], mdp.stem_type_offset[t[mol.blockidxs[i[1]]]] + i[3]) for i in mol.jbonds]
    
    # Here stem_type_offset is a list of offsets to know which embedding to use for a particular stem. 
    # Each (blockidx, atom) pair has its own embedding.
    stemtypes = [mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[1] for i in mol.stems]

    data = Data(
        x=f([t[i] for i in mol.blockidxs]),
        edge_index=f(edges).T if len(edges) else f([[],[]]),
        edge_attr=f(edge_attrs) if len(edges) else f([]).reshape((0,2)),
        stems=f(mol.stems) if len(mol.stems) else f([(0,0)]),
        stemtypes=f(stemtypes) if len(mol.stems) else f([mdp.num_stem_types])
    )
    data.to(mdp.device)
    
    assert not bonds and not nblocks

    return data

def mols2batch(mols, mdp):
    batch = Batch.from_data_list(mols, follow_batch=['stems'])
    batch.to(mdp.device)
    return batch





class GraphAgent_rwd(nn.Module):
    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, version='v1'):
        super().__init__()

        self.v1_model = StateEmbeddingNet(2*nemb, nvec, num_conv_steps, mdp_cfg, version)
        self.global2pred = nn.Sequential(
            nn.Linear(4*nemb, 4*nemb),
            nn.LeakyReLU(), 
            nn.Linear(4*nemb, 1)
        )

    def forward(self, graph_data, vec_data=None, do_stems=True):
        emd = self.v1_model(copy.deepcopy(graph_data))
        
        global_mean_pool_out = torch.cat([emd[:-1], emd[1:]], dim = -1)
        mol_preds = self.global2pred(global_mean_pool_out)
        return mol_preds
    

class GraphAgent_model(nn.Module):
    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, version='v1'):
        super().__init__()

        self.v1_model = StateEmbeddingNet(2*nemb, nvec, num_conv_steps, mdp_cfg, version)
        self.global2pred = nn.Sequential(
            nn.Linear(2*nemb, 2*nemb),
            nn.LeakyReLU(), 
            nn.Linear(2*nemb, 1)
        )

    def forward(self, graph_data, vec_data=None, do_stems=True):
        emd = self.v1_model(copy.deepcopy(graph_data))
        mol_preds = self.global2pred(emd)
        return mol_preds
    
