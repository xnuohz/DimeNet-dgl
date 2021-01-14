import argparse
import torch
import dgl
import numpy as np

from qm9 import QM9
from modules.dimenet import DimeNet
from torch.utils.data import DataLoader

# create collate_fn
def _collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return g, labels

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

    # for g, labels in dataloader:
    g, label = dataset[3]
    logits = model(g)
    print(g, label)

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