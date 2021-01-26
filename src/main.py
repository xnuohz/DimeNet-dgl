import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl

from time import time
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from sklearn.metrics import mean_absolute_error
from qm9 import QM9
from modules.dimenet import DimeNet

@torch.no_grad()
def ema(ema_model, model, decay):
    msd = model.state_dict()
    for k, ema_v in ema_model.state_dict().items():
        model_v = msd[k].detach()
        ema_v.copy_(ema_v * decay + (1. - decay) * model_v)

def edge_init(edges):
    R_src, R_dst = edges.src['R'], edges.dst['R']
    dist = torch.sqrt(F.relu(torch.sum((R_src - R_dst) ** 2, -1)))
    # d: bond length, o: bond orientation
    return {'d': dist, 'o': R_src - R_dst}

def _collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return g, labels

def train(device, model, opt, loss_fn, train_loader):
    model.train()
    epoch_loss = 0
    num_samples = 0

    for g, labels in train_loader:
        t = time()
        g = g.to(device)
        labels = labels.to(device)
        logits = model(g)
        loss = loss_fn(logits, labels.view([-1, 1]))
        epoch_loss += loss.data.item() * len(labels)
        num_samples += len(labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return epoch_loss / num_samples

@torch.no_grad()
def evaluate(device, model, valid_loader):
    model.eval()
    predictions_all, labels_all = [], []
    
    for g, labels in valid_loader:
        g = g.to(device)
        logits = model(g)
        labels_all.extend(labels)
        predictions_all.extend(logits.view(-1,).cpu().numpy())
    
    return np.array(predictions_all), np.array(labels_all)

@profile
def main():
    # load data
    t = time()
    dataset = QM9(label_keys=args.targets, edge_funcs=[edge_init])
    print('Loading dataset... ', time() - t)
    # data split
    train_data, valid_data, test_data = split_dataset(dataset, random_state=42)
    # data loader
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=_collate_fn,
                              num_workers=args.num_workers)

    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=_collate_fn,
                              num_workers=args.num_workers)

    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=_collate_fn,
                             num_workers=args.num_workers)
    
    print('train size: ', len(train_data))
    print('valid size: ', len(valid_data))
    print('test size: ', len(test_data))

    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # model initialization
    t = time()
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
                    num_targets=len(args.targets)).to(device)
    print('Model init... ', time() - t)
    # define loss function and optimization
    loss_fn = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    # model training
    best_mae = 1e9
    no_improvement = 0
    training_times = []
    # EMA for valid and test
    t = time()
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    print('EMA init... ', time() - t)
    
    best_model = copy.deepcopy(ema_model)

    for i in range(args.epochs):
        t = time()
        train_loss = train(device, model, opt, loss_fn, train_loader)
        training_times.append(time() - t)
        ema(ema_model, model, args.ema_decay)
        predictions, labels = evaluate(device, ema_model, valid_loader)

        cur_mae = mean_absolute_error(labels, predictions)
        print('Epoch {} | Train Loss {:.4f} | Val MAE {:.4f}'.format(i, train_loss, cur_mae))

        if cur_mae > best_mae:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print('Early stop.')
                break
        else:
            no_improvement = 0
            best_mae = cur_mae
            best_model = copy.deepcopy(ema_model)

    # model testing
    predictions, labels = evaluate(device, ema_model, test_loader)
    test_mae = mean_absolute_error(labels, predictions)
    print('Training times: ', np.mean(training_times))
    print('Test MAE {:.4f}'.format(test_mae))

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
    parser.add_argument('--targets', nargs='+', type=str, help='List of targets to predict.')

    parser.set_defaults(targets=['mu'])
    # training params
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=800, help='Training epochs.')
    parser.add_argument('--early-stopping', type=int, default=20, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of subprocesses to use for data loading.')

    args = parser.parse_args()
    print(args)

    main()