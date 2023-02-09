import math
from rdflib import Graph

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.ops as F
import dgl.function as fn

from dgl.nn.pytorch import GraphConv, GATConv

from torchdiffeq import odeint

from torch.autograd import Variable

class GraphGRUODE(nn.Module):
    
    def __init__(self, in_dim, hid_dim, device=th.device('cpu'), gnn='GATConv', bias=True, **kwargs):
    
        super(GraphGRUODE, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.device = device
        self.gnn = gnn
        self.bias = bias
        self.dropout = nn.Dropout(0.1)

        if self.gnn == 'GCNConv':
            # self.lin_xx = GCNConv(self.in_dim+self.hid_dim, self.hid_dim, bias=self.bias)
            # self.lin_hx = nn.Linear(self.hid_dim, self.in_dim, bias=self.bias)
            self.lin_xz = GraphConv(self.in_dim,   self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_xr = GraphConv(self.in_dim,   self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_xh = GraphConv(self.in_dim,   self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_hz = GraphConv(self.hid_dim,  self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_hr = GraphConv(self.hid_dim,  self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_hh = GraphConv(self.hid_dim,  self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
        elif self.gnn == 'GATConv':
            self.lin_xz = GATConv(self.in_dim,  self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_xr = GATConv(self.in_dim,  self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_xh = GATConv(self.in_dim,  self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_hz = GATConv(self.hid_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_hr = GATConv(self.hid_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_hh = GATConv(self.hid_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
        else:
            raise NotImplementedError

        self.edge_index = None
        self.x = None

    #     self.reset_parameters()

    # def reset_parameters(self):

    #     self.lin_xz.reset_parameters()

    def set_graph(self, graph: dgl.DGLGraph):
    
        self.graph = graph

    def set_x(self, x): 
        self.x = x.to(self.device)

    def forward(self, t, h):

        # x = torch.zeros_like(h).to(self.device)

        # edge_index = self.edge_index_batchs[0]
        
        # rint(t)
        
        node_idx   = self.graph.filter_nodes(lambda nodes: nodes.data['t'] >= t) # filter out sessions already computed 
        edge_idx   = self.graph.filter_edges(lambda edges: edges.data['t'] <= t) 
        edge_index = (self.graph.edges()[0][edge_idx], self.graph.edges()[1][edge_idx])
        # edge_index = self.graph.edges()
        graph      = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.graph.number_of_nodes(), device=self.device)
        graph      = dgl.node_subgraph(graph, node_idx)
        graph      = dgl.remove_self_loop(graph)
        graph      = dgl.add_reverse_edges(graph)
        # graph      = dgl.add_self_loop(graph)
        # print(graph.number_of_nodes(), graph.number_of_edges())
        # graph      = dgl.add_self_loop(graph)
        # graph = self.graph
        # x = self.dropout(self.x)
        # h = self.dropout(h)
        ht = h[node_idx]
        x  = self.x[node_idx]
        
        # print(len(x), graph.num_nodes())
        
        Dh = th.zeros_like(self.x)

        # print(Dh.shape, )

        if self.gnn == 'GATConv':
            # x = self.lin_xx(torch.cat((self.x.to(self.device), h), dim=1), edge_index).to(self.device)
            xr, xz, xh = self.lin_xr(graph, x).max(1)[0], self.lin_xz(graph, x).max(1)[0], self.lin_xh(graph, x).max(1)[0]
            r = th.sigmoid(xr + self.lin_hr(graph,  ht).max(1)[0])
            z = th.sigmoid(xz + self.lin_hz(graph,  ht).max(1)[0])
            u = th.tanh(xh + self.lin_hh(graph, r * ht).max(1)[0])
            dh = (1 - z) * (u - ht)
        elif self.gnn == 'GCNConv':
            xr, xz, xh = self.lin_xr(graph, x), self.lin_xz(graph, x), self.lin_xh(graph, x)
            r = th.sigmoid(xr + self.lin_hr(graph,  ht))
            z = th.sigmoid(xz + self.lin_hz(graph,  ht))
            u = th.tanh(xh + self.lin_hh(graph, r * ht))
            dh = (1 - z) * (u - ht)
        elif self.gnn == 'Linear':
            # print(h.shape)
            # h = self.lin_hx(h)+self.lin_xx(x)
            # x = self.propagate(edge_index=edge_index, x=h, aggr='mean')-h
            xr, xz, xh = self.lin_xr(x), self.lin_xz(x), self.lin_xh(x)
            r = th.sigmoid(xr  + self.lin_hr(ht))
            z = th.sigmoid(xz  + self.lin_hz(ht))
            u = th.tanh(xh + self.lin_hh(r * ht))
            dh = (1 - z) * (u - ht)
        elif self.gnn == 'MLP':
            dh = self.enc(ht)
        Dh[node_idx] = dh
        # dh = nn.functional.normalize(dh)
        # self.x = self.hx(dh, edge_index)
        return Dh

class GGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, feat_drop=0.0, activation=None):
        super().__init__()
        self.dropout    = nn.Dropout(feat_drop)
        self.gru        = nn.GRUCell(2 * output_dim, input_dim)
        self.W1         = nn.Linear(input_dim, output_dim, bias=False)
        self.W2         = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation
        
    def messager(self, edges):
        
        return {'m': edges.src['ft'] * edges.data['w'].unsqueeze(-1), 'w': edges.data['w']}

    def reducer(self, nodes):
        
        m = nodes.mailbox['m']
        w = nodes.mailbox['w']
        hn = m.sum(dim=1) / w.sum(dim=1).unsqueeze(-1)
        
        return {'neigh': hn}
    
    def forward(self, mg, feat):
        with mg.local_scope():
            mg.ndata['ft'] = self.dropout(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(self.messager, self.reducer)
                neigh1 = mg.ndata['neigh']
                mg1 = mg.reverse(copy_edata=True)
                mg1.update_all(self.messager, self.reducer)
                neigh2 = mg1.ndata['neigh']
                neigh1 = self.W1(neigh1)
                neigh2 = self.W2(neigh2)
                hn = th.cat((neigh1, neigh2), dim=1)
                rst = self.gru(hn, feat) 
            else:
                rst = feat
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

class AttnReadout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v)
        e = self.fc_e(th.sigmoid(feat_u + feat_v)) 
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e) 
        feat_norm = feat * alpha
        rst = F.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

class GNG_ODE(nn.Module):
    
    def __init__(self, name, num_items, gnn, embedding_dim, num_layers, feat_drop=0.0, norm=True, scale=12, solver=None, num_splits=1):
        super().__init__()
        self.name      = name
        self.gnn       = gnn
        self.num_items = num_items
        self.num_splits = num_splits
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.register_buffer('indices', th.arange(num_items, dtype=th.long))
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norm = norm
        self.scale = scale
        self.solver = solver
        input_dim = embedding_dim
        print(name)
        # print(name + "**********************************************************")
        for i in range(num_layers):
            layer = GGNNLayer(
                input_dim,
                embedding_dim * 2 if name == "tmall" else embedding_dim,
                feat_drop=feat_drop
            )
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim,
            embedding_dim,
            embedding_dim,
            batch_norm=None,
            feat_drop=feat_drop,
            activation=None,
        )
        
        self.ODEFunc = GraphGRUODE(self.embedding_dim, self.embedding_dim, gnn=gnn, device=th.device('cuda:0'))
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(input_dim + embedding_dim, embedding_dim, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, mg, embeds_ids, times, num_nodes):
        
        iid = mg.ndata['iid']
        
        # print(iid.max(), self.num_items)
        feat = self.feat_drop(self.embedding(iid))
        if self.norm:
            feat = nn.functional.normalize(feat) # feat.div(th.norm(feat, p=2, dim=-1, keepdim=True) + 1e-12)
        
        # feat0 = feat
        out   = feat
        for i, layer in enumerate(self.layers):
            out = layer(mg, out)
        
        feat = out
        
        if self.norm:
            feat = nn.functional.normalize(feat)
        
        self.ODEFunc.set_graph(mg)
        self.ODEFunc.set_x(feat)
        # print(mg.edata)
        t_end  = mg.edata['t'].max() ## max time in the batch
        # step_t = th.unique(mg.edata['t'].sort()[0]
        t      = th.tensor([0., t_end], device=mg.device) 
        # t_end  = mg.edata['t']
        # times = mg.edata['t'].sort()
        # sort_times = th.unique(mg.edata['t'])
       
        if self.solver != "dopri5":
            feat = odeint(self.ODEFunc, feat, t=t, method=self.solver, options={"perturb": "True", "step_size": t_end / self.num_splits})[-1] # .mean(0)
        else:
            # feat = odeint(self.ODEFunc, feat, t=t, rtol=1e-1, atol=1e-2, options={"first_step": 0.2})[-1]
            feat = odeint(self.ODEFunc, feat, t=t, rtol=1e-4, atol=1e-5)[-1]# , options={"first_step": 0.0})[-1]
            
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        if self.norm:
            feat = feat.div(th.norm(feat, p=2, dim=-1, keepdim=True))
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(sr)
        if self.norm:
            sr = sr.div(th.norm(sr, p=2, dim=-1, keepdim=True) + 1e-12)
        target = self.embedding(self.indices)
        if self.norm:
            target = target.div(th.norm(target, p=2, dim=-1, keepdim=True) + 1e-12)
        logits = sr @ target.t()
        if self.scale:
            logits = th.log_softmax(self.scale * logits, dim=-1)
        else:
            logits = th.log_softmax(logits, dim=-1)
        return logits# , kl_loss # , 0
        
        
        