# TAAF-D-GEX
D-GEX with transformative adaptive activation functions (TAAFs)



## Motivation
Gene expression profiling was made cheaper by the NIH LINCS program that profiles
only 1000 selected landmark genes and uses them to reconstruct the whole profile. The [D-GEX](https://github.com/uci-cbcl/D-GEX)
method employs neural networks to infer the whole profile. However, the original D–GEX can be further
significantly improved.

## Results
We have analyzed the D–GEX method and determined that the inference can be improved using
a logistic sigmoid activation function instead of the hyperbolic tangent. Moreover, we propose a novel
transformative adaptive activation function that improves the gene expression inference even further and
which generalizes several existing adaptive activation functions. Our improved neural network achieve
average mean absolute error of **0.1340** which is a significant improvement over our reimplementation of
the original D–GEX which achieves average mean absolute error **0.1637**.

## Data
The used data are the same as in [D-GEX](https://github.com/uci-cbcl/D-GEX) but
are not provided here due to their size. The code used for preprocessing the data
is  in folder `dataset_preparation`. Very small subset of the used dataset is provided
for running the example.

## Example
To quickly train your own network with TAAF you use our example. The
`example.py` and `example_classical.py` provide code necessary for
training small network for Gene Expression inference with and without
TAAF.

### Example Data
The example data represent a small subset of inferred target gene with
 much less samples than used in the main paper. It uses **942 landmark genes**
 to reconstruct **952 of randomly selected target genes** (10 %
 of target genes used in the paper).

### Example architectures
 The network used in example somewhat resemble the networks used in the
 paper - they consists of 3 dense layer (one with TAAFs, the other
 without). The main difference is that the network consists of only
 300 neurons in each layer and thus create a bottleneck (which hampers
 the performance) in order to allow training the example quickly even
 without GPU acceleration or using a laptop GPU.

#### Classical network
![Visualization of architecture with classical activation functions](https://github.com/kunc/TAAF-D-GEX/tree/master/models/classical_demonstrator_small/classical_demonstrator_small_visualization.png "[Visualization of architecture with classical activation functions")

```
Layer (type)                 Output Shape              Param #
=================================================================
inputs (InputLayer)          (None, 942)               0
_________________________________________________________________
H0 (Dense)                   (None, 300)               282900
_________________________________________________________________
dropout_1 (Dropout)          (None, 300)               0
_________________________________________________________________
H1 (Dense)                   (None, 300)               90300
_________________________________________________________________
dropout_2 (Dropout)          (None, 300)               0
_________________________________________________________________
outputs (Dense)              (None, 952)               286552
=================================================================
Total params: 659,752
Trainable params: 659,752
Non-trainable params: 0
_________________________________________________________________

```

#### TAAF network
![Visualization of architecture with TAAFs](https://github.com/kunc/TAAF-D-GEX/tree/master/models/TAAF_demonstrator_small/TAAF_demonstrator_small_visualization.png "[Visualization of architecture with TAAFs")

```
Layer (type)                 Output Shape              Param #
=================================================================
inputs (InputLayer)          (None, 942)               0
_________________________________________________________________
H0_Dense (Dense)             (None, 300)               282600
_________________________________________________________________
H0_AdaptiveTAAF_Bottom (ATU) (None, 300)               600
_________________________________________________________________
activation_1 (Activation)    (None, 300)               0
_________________________________________________________________
H0_AdaptiveTAAF_Top (ATU)    (None, 300)               600
_________________________________________________________________
dropout_1 (Dropout)          (None, 300)               0
_________________________________________________________________
H1_Dense (Dense)             (None, 300)               90000
_________________________________________________________________
H1_AdaptiveTAAF_Bottom (ATU) (None, 300)               600
_________________________________________________________________
activation_2 (Activation)    (None, 300)               0
_________________________________________________________________
H1_AdaptiveTAAF_Top (ATU)    (None, 300)               600
_________________________________________________________________
dropout_2 (Dropout)          (None, 300)               0
_________________________________________________________________
outputs_Dense (Dense)        (None, 952)               285600
_________________________________________________________________
outputs_AdaptiveTAAF_Bottom  (None, 952)               1904
_________________________________________________________________
activation_3 (Activation)    (None, 952)               0
_________________________________________________________________
outputs_AdaptiveTAAF_Top (AT (None, 952)               1904
=================================================================
Total params: 664,408
Trainable params: 664,408
Non-trainable params: 0
_________________________________________________________________

```

### Results
The results are not as completely presented as in the paper as here
only training and testing sets are used, the main paper uses also the
validation test for selection of the best performing network without
introducing bias to the test results. Here only two sets are used thus
we present the results for all epochs and the minimum reached during
on the test data.

![MAE during training](https://github.com/kunc/TAAF-D-GEX/tree/master/example_data/figures/MAE.png "[MAE during training")

The plot above shows performance of both network during the whole
training (the MAE on the training data is dashed and on testing data solid).
Both networks overfitted eventhough the network with TAAFs overfitted
much more (due to increased capacity of the network due to presence of
TAAFs and slightly more weights). If we selected the minimum loss on
the unseen data, we can see that the network with TAAFs outperforms the
classical network.


| Network   | min. MAE (train) | min. MAE (test) |
| ----------|:----------------:|----------------:|
| TAAF      | **0.17866**      | **0.19068**     |
| classical | 0.19045          |   0.20115       |