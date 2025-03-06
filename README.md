# CubeDimAE


### Objective

This algorithm aims to estimate the intrinsic dimension of a dataset to reduce the wasted effort finding the proper size of the latent space in designing an autoencoder.


### Overview

(overview diagram)


### Grid Orientation

This method is not intrinsically rotation-invariant. Instead, it uses the PCA to align the axes along the greatest variances, to stabilize the connections and ensure the rotation-invariance of the result.
