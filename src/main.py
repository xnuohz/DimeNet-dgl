import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from sklearn.metrics import mean_absolute_error
from qm9 import QM9
from modules.dimenet import DimeNet
from utils import _collate_fn

def train(device, model, opt, loss_fn, train_loader):
    model.train()
    epoch_loss = 0
    for g, labels in train_loader:
        g = g.to(device)
        labels = labels.to(device)
        logits = model(g)
        loss = loss_fn(logits, labels.view([-1, 1]))
        epoch_loss += loss.data.item() * len(labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return epoch_loss / len(train_loader)

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

def main():
    # load data
    dataset = QM9(label_keys=args.targets)
    # data split
    train_data, valid_data, test_data = split_dataset(dataset, random_state=42)
    # data loader
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=_collate_fn)

    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=_collate_fn)

    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=_collate_fn)
    
    print('train size: ', len(train_data))
    print('valid size: ', len(valid_data))
    print('test size: ', len(test_data))

    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # model initialization
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
                    num_targets=len(args.targets))
        
    model = model.to(device)
    # define loss function and optimization
    loss_fn = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # model training
    best_mae = 1e9
    best_model = copy.deepcopy(model)
    no_improvement = 0

    for i in range(args.epochs):
        train_loss = train(device, model, opt, loss_fn, test_loader)
        predictions, labels = evaluate(device, model, test_loader)

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
            best_model = copy.deepcopy(best_model)

    # model testing
    predictions, labels = evaluate(device, best_model, test_loader)
    test_mae = mean_absolute_error(labels, predictions)
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
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3000000, help='Training epochs.')
    parser.add_argument('--early-stopping', type=int, default=20, help='Patient epochs to wait before early stopping.')

    args = parser.parse_args()
    print(args)

    main()