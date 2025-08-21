# CubeDimAE



### Introduction


A proof-of-concept(PoC) paper aims to reduce the wasted effort finding the proper size of the latent space in designing an autoencoder, by estimating in advance the intrinsic dimension of a dataset.
The estimation process was inspired by how human interpolates discontinuous set of points and imagine a continuous analogue in his/her mind.
This is in contrast to the traditional approaches, which first assume the existence of some continuous manifold, seeing the dataset at hand as a finite sample from the infinite set,
and try to guess the dimension of the manifold by examining various statistical properties of the sample.



### Files


 - **cubedimae.pdf**: The paper published at **BigComp2025** on February.
 - **cube_dim.py**: The implementation of the algorithm.
 - *experiment.py*: The experiment script used for the paper.
 - requirement.txt: The packages used for the implementation and experiment.



&nbsp;
&nbsp;
&nbsp;
## Algorithm



### Overview



To describe the algorithm in an intuitive manner,

1. Introduce a cubic grid on the data space.
2. "Color" the non-empty regions.
4. Count the adjacent cubes for every cube.
5. If the average count is near `3 ** k - 1`, we conclude the intrinsic dimension is *k*.

The rationale behind the expression is described in the paper in detail.


!['readme_figures/overview.png' not found](readme_figures/overview.png)



### Datasets



### Evaluation
