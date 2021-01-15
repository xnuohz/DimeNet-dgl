import argparse
import torch
import dgl
import numpy as np
import scipy.sparse as sp

from qm9 import QM9
from modules.dimenet import DimeNet
from torch.utils.data import DataLoader

# create collate_fn
def _collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    
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

def main():
    dataset = QM9(cutoff=args.cutoff, label_keys=['mu'])
    
    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)

    model = DimeNet(emb_size=args.emb_size,
                    num_blocks=args.num_blocks,
                    num_bilinear=args.num_bilinear,
                    num_spherical=args.num_spherical,
                    num_radial=args.num_radial,
                    cutoff=args.cutoff,
                    envelope_exponent=args.envelope_exponent,
                    num_before_skip=args.num_before_skip,
                    num_after_skip=args.num_after_skip,
                    num_dense_output=args.num_dense_output,
                    num_targets=args.num_targets)

    for g, data, labels in dataloader:
        logits = model(g, data)
        break

if __name__ == "__main__":
    """
    DimeNet Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='DimeNet')

    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    # model params
    parser.add_argument('--emb-size', type=int, default=128, help='Embedding size used throughout the model.')
    parser.add_argument('--num-blocks', type=int, default=6, help='Number of building blocks to be stacked.')
    parser.add_argument('--num-bilinear', type=int, default=8, help='Third dimension of the bilinear layer tensor.')
    parser.add_argument('--num-spherical', type=int, default=7, help='Number of spherical harmonics.')
    parser.add_argument('--num-radial', type=int, default=6, help='Number of radial basis functions.')
    parser.add_argument('--envelope-exponent', type=int, default=5, help='Shape of the smooth cutoff.')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff distance for interatomic interactions.')
    parser.add_argument('--num-before-skip', type=int, default=1, help='Number of residual layers in interaction block before skip connection.')
    parser.add_argument('--num-after-skip', type=int, default=2, help='Number of residual layers in interaction block after skip connection.')
    parser.add_argument('--num-dense-output', type=int, default=3, help='Number of dense layers for the output blocks.')
    parser.add_argument('--num-targets', type=int, default=12, help='Number of targets to predict.')

    args = parser.parse_args()
    print(args)

    main()