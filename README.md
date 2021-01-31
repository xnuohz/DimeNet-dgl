# DGL Implementation of DimeNet

This DGL example implements the GNN model proposed in the paper [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123). For the original implementation, see [here](https://github.com/klicperajo/dimenet).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
dgl 0.5.2
numpy 1.19.4
pandas 1.1.4
tqdm 4.53.0
torch 1.7.0
sympy 1.7.1
scikit-learn 0.23.2
click 7.1.2
logzero 1.6.3
ruamel.yaml 0.16.12
```

### The graph datasets used in this example

The DGL's built-in QM9 dataset. Dataset summary:

* Molecular Graphs: 13,0831
* Number of Tasks: 12

### Usage

###### GPU options
```
gpu               int   GPU index.                Default is -1, using CPU.
```

###### Model options
```
emb-size          int   Embedding size used throughout the model.                              Default is 128
num-blocks        int   Number of building blocks to be stacked.                               Default is 6   
num-bilinear      int   Third dimension of the bilinear layer tensor.                          Default is 8   
num-spherical     int   Number of spherical harmonics.                                         Default is 7   
num-radial        int   Number of radial basis functions.                                      Default is 6   
envelope-exponent int   Shape of the smooth cutoff.                                            Default is 5   
cutoff            float Cutoff distance for interatomic interactions.                          Default is 5.0 
num-before-skip   int   Number of residual layers in interaction block before skip connection. Default is 1   
num-after-skip    int   Number of residual layers in interaction block after skip connection.  Default is 2   
num-dense-output  int   Number of dense layers for the output blocks.                          Default is 3   
targets           list  List of targets to predict.                                            Default is ['mu']
```

###### Training options
```
lr                float Learning rate.                                  Default is 0.001
weight-decay      float Weight decay.                                   Default is 0.0001
ema-decay         float EMA decay.                                      Default is 0.999
batch-size        int   Batch size.                                     Default is 32
epochs            int   Training epochs.                                Default is 800
early-stopping    int   Patient epochs to wait before early stopping.   Default is 20
num-workers       int   Number of subprocesses to use for data loading. Default is 0
```

###### Examples

The following commands learn a neural network and predict on the test set.
Training a DimeNet++ model on QM9 dataset.
```bash
python src/main.py --model-cnf src/config/dimenet_pp.yaml
```

### Performance

| Target | mu | alpha | homo | lumo | gap | r2 | zpve | U0 | U | H | G | Cv |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| MAE(DimeNet in Table 1) | 0.0286 | 0.0469 | 27.8 | 19.7 | 34.8 | 0.331 | 1.29 | 8.02 | 7.89 | 8.11 | 8.98 | 0.0249 |
| MAE(DGL) | 2.6134 | 49.40 | 4.88 | 1.03 |  |  |  |  | 49.1 | 47.6 | 41.1 | 23.653 |

### Speed

| Model | Original Implementation | DGL Implementation | Improvement |
| :-: | :-: | :-: | :-: |
| DimeNet | 2839 | 2940 | |