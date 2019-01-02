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
The used are the same as in [D-GEX](https://github.com/uci-cbcl/D-GEX) but
are not provided here due to their size. The code used for preprocessing the data
is  in folder `dataset_preparation`. Very small subset of the used dataset is provided
for running the example.