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
```

### The graph datasets used in this example

The DGL's built-in QM9 dataset. Dataset summary:

* Molecular Graphs: 13,0831
* Number of Tasks: 12

### Usage

###### GPU options
```
--gpu       int   GPU index.                Default is -1, using CPU.
```

###### Model options
```
--emb-size          int     Embedding size used throughout the model.                              Default is 128.
--num-blocks        int     Number of building blocks to be stacked.                               Default is 6   
--num-bilinear      int     Third dimension of the bilinear layer tensor.                          Default is 8   
--num-spherical     int     Number of spherical harmonics.                                         Default is 7   
--num-radial        int     Number of radial basis functions.                                      Default is 6   
--envelope-exponent int     Shape of the smooth cutoff.                                            Default is 5   
--cutoff            float   Cutoff distance for interatomic interactions.                          Default is 5.0 
--num-before-skip   int     Number of residual layers in interaction block before skip connection. Default is 1   
--num-after-skip    int     Number of residual layers in interaction block after skip connection.  Default is 2   
--num-dense-output  int     Number of dense layers for the output blocks.                          Default is 3   
--num-targets       int     Number of targets to predict.                                          Default is 12  
```

###### Examples

The following commands learn a neural network and predict on the test set.
Training a DimeNet model on QM9 dataset.
```bash
python main.py
```

### Performance

| Target | mu | alpha | homo | lumo | gap | r2 | zpve | U0 | U | H | G | Cv |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| MAE(DimeNet in Table 1) | 0.0286 | 0.0469 | 27.8 | 19.7 | 34.8 | 0.331 | 1.29 | 8.02 | 7.89 | 8.11 | 8.98 | 0.0249 |
| MAE(DGL) |  |  |  |  |  |  |  |  |  |  |  |  |

### QA

* 2-hop angle, node's src and dst in line_graph