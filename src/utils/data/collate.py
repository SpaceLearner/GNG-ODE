from collections import Counter
import numpy as np
import torch as th
import torch.nn.functional as F
import dgl
import pickle


def label_last(g, last_nid):
    is_last = th.zeros(g.num_nodes(), dtype=th.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g

def label_last_ccs(g, last_nid):
    for i in range(len(last_nid)):
        is_last = th.zeros(g.num_nodes('s'+str(i+1)), dtype=th.int32)
        is_last[last_nid[i]] = 1
        g.nodes['s'+str(i+1)].data['last'] = is_last
    return g

def label_last_k(g, last_nids):
    is_last = th.zeros(g.number_of_nodes(), dtype=th.int32)
    is_last[last_nids] = 1
    g.nodes['s1'].data['last'] = is_last
    return g

def seq_to_eop_multigraph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
    else:
        src = th.LongTensor([])
        dst = th.LongTensor([])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])
    return g

def seq_to_shortcut_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    edges = counter.keys()
    src, dst = zip(*edges)

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g

def seq_to_session_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]
    )
    edges = counter.keys()
    if len(edges) > 0:
        src, dst = zip(*edges)
        weight = th.tensor(list(counter.values()))
    else:
        src, dst = [0], [0]
        weight = th.ones(1).long()

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    
    g.edata['w'] = weight
    # print(g.edata)
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])

    return g

def seq_to_temporal_session_graph(seq, times):
    items, indices = np.unique(seq, return_index=True)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    # counter = Counter(
    #     [(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]
    # )
    edges = [[seq_nid[i], seq_nid[i+1]] for i in range(len(seq)-1)]
    # edges = counter.keys()
    
    if len(edges) > 0:
        src, dst = zip(*edges)
        weight = th.ones(len(edges)).long()
    else:
        src, dst = [0], [0]
        weight = th.ones(1).long()

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    # print(len(edges), g.number_of_edges())
    g.edata['w']  = weight
    # print(len(times), g.number_of_nodes())
    # g.ndata['t']  = th.tensor(times)[indices]
    g.ndata['t'] = th.ones(g.num_nodes()) * max(times)
    # print(g.edata, times, g.number_of_edges(), g.number_of_nodes())
    if g.number_of_edges() == 1 and g.number_of_nodes() == 1:
        g.edata['t'] = th.tensor(times)[0].unsqueeze(-1)
    else:
        g.edata['t'] = th.tensor(times)[1:] # [-g.number_of_edges():]
    
    # print(g.edata)
    
    g.ndata['iid'] = th.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])

    return g
    
def collate_fn_factory(*seq_to_graph_fns):
    def collate_fn(samples):
        seqs, labels = zip(*samples)
        inputs = []
        for seq_to_graph in seq_to_graph_fns:
            graphs = list(map(seq_to_graph, seqs))        
            bg = dgl.batch(graphs)
            inputs.append(bg)
        labels = th.LongTensor(labels)
        return inputs, labels

    return collate_fn

def collate_fn_factory_temporal(*seq_to_graph_fns):
    def collate_fn(samples):
        seqs, times, labels = zip(*samples)
        inputs     = []
        for seq_to_graph in seq_to_graph_fns:
            graphs    = list(map(seq_to_graph, seqs, times))
            num_nodes = th.tensor([graph.number_of_nodes() for graph in graphs], dtype=th.long) 
            max_num   = max(num_nodes)
            embeds_id = th.vstack([F.pad(graph.ndata['iid'], (0, max_num-len(graph.ndata['iid'])), value=graph.ndata['iid'][-1]) for graph in graphs])
            times     = th.vstack([F.pad(graph.ndata['t'],   (0, max_num-len(graph.ndata['iid'])), value=graph.ndata['iid'][-1]) for graph in graphs])
            bg        = dgl.batch(graphs)
            inputs.append(bg)
        labels = th.LongTensor(labels)
        # print(inputs[0].edata)
        return inputs, labels, embeds_id, times, num_nodes

    return collate_fn

if __name__ == '__main__':
    
    # seq = [3, 1, 3, 6, 2, 5, 1, 2, 4, 1, 2] # 2, 0, 2, 5, 1, 4, 0, 1, 3, 0, 1 
    # seq0 = [250, 250, 250, 250, 3, 1, 2, 4, 1]
    # # g1 = seq_to_ccs_graph(seq, order=4)
    # # g2 = seq_to_ccs_graph(seq, order=2)
    # collate_fn = collate_fn_factory_ccs(seq_to_ccs_graph, order=2)
    # seqs = [[seq, 1], [seq0, 2]]
    # print(collate_fn(seqs)[0][0].batch_num_nodes('s2'))
    seq1  = [3, 1, 3, 6, 2, 5, 1, 2, 4, 1, 2]
    time1 = [0, 1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9]
    
    g1 = seq_to_mis_graph(seq1, time1, num_inters=3)
    
    seq2  = [3, 1, 3, 6, 2, 5, 1, 2, 4, 1, 2]
    time2 = [0, 1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9]
    
    g2 = seq_to_mis_graph(seq2, time2, num_inters=3)
    
    seqs = [[seq1, time1, 1], [seq2, time2, 2]]
    
    collate_fn = collate_fn_factory_mis(seq_to_mis_graph, num_inters=3)
    
    print(collate_fn(seqs)[0][0].nodes["interest"])
    
