# GS Skew Orthogonal Convolutions
This directory contains all the code for orthogonal convolutions experiments with variations of accelerating basic Skew Orthogonal Convolution.


## Installation

In order to install necessary dependencies run the following command:

```
pip install -r requirements.txt
```

## How to train CNN and choose layer options

```
python3 train_*.py
```

where instead of `*` either `standard` or `robust` can be inserted. All the training parameters are managed using `hydra-core` library and stored in `conf/`.

 - Convolutional layer options: `standard, soc, gs_soc, gs_soc_accelerated, permuted_soc, lpr_soc`. See `skew_ortho_conv.py` for detatils.
 - Activation functions: `minmax, minmax_permuted`. See `custom_activations.py` for details.
