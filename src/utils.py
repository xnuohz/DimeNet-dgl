import numpy as np
import torch
import torch.nn.functional as F
import dgl

def edge_init(edges):
    R_src, R_dst = edges.src['R'], edges.dst['R']
    dist = torch.sqrt(F.relu(torch.sum((R_src - R_dst) ** 2, -1)))
    # d: bond length, o: bond orientation
    return {'d': dist, 'o': R_src - R_dst}

# create collate_fn
def _collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    g.apply_edges(edge_init)

    labels = torch.tensor(labels, dtype=torch.float32)
    return g, labels