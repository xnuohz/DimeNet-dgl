import dgl
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

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

    data = {}

    dst, src = g.edges()
    edgeid_to_target, edgeid_to_source = dst.numpy(), src.numpy()

    # Target (i) and source (j) nodes of edges
    data['idnb_i'] = dst.type(torch.long)
    data['idnb_j'] = src.type(torch.long)

    # Indices of triplets k->j->i
    adj_matrix = g.adj(scipy_fmt='csr')

    degree = g.out_degrees().numpy()
    ntriplets = degree[edgeid_to_source]
    id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
    id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
    id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]

    # Indices of triplets that are not i->j->i
    id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
    data['id3dnb_i'] = torch.LongTensor(id3ynb_i[id3_y_to_d])
    data['id3dnb_j'] = torch.LongTensor(id3ynb_j[id3_y_to_d])
    data['id3dnb_k'] = torch.LongTensor(id3ynb_k[id3_y_to_d])

    atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape)
    
    # Edge indices for interactions
    # j->i => k->j
    data['id_expand_kj'] = torch.LongTensor(atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d])
    # j->i => k->j => j->i
    data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
    
    N = len(g.ndata['Z'])
    data['batch_seg'] = np.repeat(np.arange(len(batch)), N)

    labels = torch.tensor(labels, dtype=torch.float32)
    return g, data, labels

def mae_loss(predictions, labels):
    return torch.abs(predictions - labels).mean()