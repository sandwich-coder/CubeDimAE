# CubeDimAE



## Introduction


A proof-of-concept(PoC), this paper aims to reduce the wasted effort finding the proper size of the latent space in designing an autoencoder by estimating in advance the intrinsic dimension of a dataset. The estimation process was inspired by how human interpolates discontinuous set of points and imagine a continuous analogue. This is contrary to the traditional approaches, which first assume the existence of some continuous manifold (manifold hypothesis), seeing the dataset at hand as a finite sample from the infinite set, and try to guess the dimension of that manifold by examining various statistical properties of the sample.

⚠️ *The scripts are supplementary and were designed for internal experiments.*



## Contribution


We propose a method to estimate the intrinsic dimension of a dataset, which is known to be the optimal latent dimension of an autoencoder, and automatically design an optimal autoencoder. We show its feasibility on five datasets.



## Files


 - **cubedimae.pdf**: The paper published at **BigComp2025** on February.
 - cube_dim.py: The implementation of the algorithm.
 - experiment.py: The experiment script used for the paper.
   - Usage: `python experiment.py`
 - requirement.txt: The packages used for the implementation and experiment.
 - readme_materials: The figures and plotting script for this README.



&nbsp;
## Algorithm Overview


To describe the algorithm in an intuitive manner,

1. Introduce a cubic grid on the data space.
2. "Color" the non-empty regions.
4. For every cube, count the adjacent cubes.
5. If the average count is near `3 ** k - 1`, we conclude the intrinsic dimension is *k*.

The rationale behind the expression is described in the paper in detail.


!['readme_materials/readme_figures/overview.png' not found](readme_materials/readme_figures/overview.png)



## Datasets


Datasets used are as follows:

1. S-curve (2-dimensional)
2. Swiss roll (2-dimensional)
3. Möbius strip (2-dimensional)
4. Hollow sphere (2-dimensional)
5. Solid sphere (3-dimensional)

They are toy datasets whose complexities, or intrinsic dimensions, we all agree on.


!['readme_materials/readme_figures/datasets.png' not found](readme_materials/readme_figures/datasets.png)



## Accuracy


The algorithm correctly estimated the dimensions of all the datasets.


<div align="center">

| Dataset | Dimension | Estimated (exact) |
| --- | --- | --- |
| S curve | 2 | 2 (2.27) |
| Swiss roll | 2 | 2 (2.26) |
| Möbius strip | 2 | 2 (2.25) |
| Hollow sphere | 2 | 2 (2.33) |
| Solid sphere | 3 | 3 (2.87) |

</div>



## Evaluation


When we do not know the proper latent dimension for the input dataset, we would try every possible values. However, we can save significant amount of time training if we know the optimal size of the bottleneck in advance.

- baseline: Trying every possible value, from 1 to 3.
- **CubeDimAE**: Estimation of the intrinsic dimension, followed by training the autoencoder *only* *once*, saving roughly 40% of time.


!['readme_materials/readme_figures/evaluation.png' not found](readme_materials/readme_figures/evaluation.png)
