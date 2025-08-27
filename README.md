<div align="center">

# CubeDimAE: Automatic Autoencoder Generation based on Dimension Estimation by Tessellation

[![Paper](https://img.shields.io/badge/paper-BigComp2025-blue.svg)](https://ieeexplore.ieee.org/document/10936902)

</div>

### Table of Contents

1. [Introduction](#introduction)
2. [Contribution](#contribution)
3. [Motivation](#motivation)
4. [Overview](#overview)
5. [Datasets](#datasets)
6. [Evaluation](#evaluation)
7. [Reproduction](#reproduction)
8. [Citation](#citation)

## Introduction

Autoencoder is the first choice in AI-based anomaly detection tasks. It has been recognized that the optimal bottleneck of an autoencoder is the intrinsic dimension of the input. Traditionally the optimal point is found through trial-and-error, putting large amount of time and effort into hyper-parameter tuning. This paper presents a method to estimate the intrinsic dimension in advance to automate the bottleneck tuning. We have tested on five synthetic datasets as a proof-of-concept and shown its feasibility with 100% accuracy and 40% reduction in time.

## Contribution

1. **Automatation of the bottleneck tuning of autoencoder network.**
2. **100% accuracy on five synthetic datasets.**
3. **40% time saving in hyper-parameter tuning compared with the baseline.**

## Motivation

Autoencoder's reconstruction performance drops drastically under the dataset's underlying complexity, which means the optimal size of the bottleneck is the dataset's intrinsic dimension. It was desirable to guess that value without trial-and-error.

<div align="center">

### S-curve Case Study

<p align="center">
  <img width="75%" src="materials/readme_figures/case_study.png">
</p>

</div>

&nbsp;
## Overview

To describe the algorithm in an intuitive manner,

1. Introduce a cubic grid on the data space.
2. "Color" the non-empty regions.
4. For every cube, count the adjacent cubes.
5. If the average count is near `3 ** k - 1`, we conclude the intrinsic dimension is *k*.

The rationale behind the expression is described in the paper in detail.

!['materials/readme_figures/overview.png' not found](materials/readme_figures/overview.png)

## Datasets

Below are datasets used in the experiments.

1. S-curve (2-dimensional)
2. Swiss roll (2-dimensional)
3. Möbius strip (2-dimensional)
4. Hollow sphere (2-dimensional)
5. Solid sphere (3-dimensional)

They are toy datasets whose complexities, or intrinsic dimensions, we all agree on.

!['materials/readme_figures/datasets.png' not found](materials/readme_figures/datasets.png)

## Evaluation

### Accuracy

The algorithm correctly estimated the dimensions of all the datasets.

<div align="center">

Dataset | Dimension | Estimated (exact)
---: | :---: | :---:
S curve | 2 | 2 (2.27)
Swiss roll | 2 | 2 (2.26)
Möbius strip | 2 | 2 (2.25)
Hollow sphere | 2 | 2 (2.33)
Solid sphere | 3 | 3 (2.87)

</div>

### Efficiency

When we do not know the proper latent dimension for the input dataset, we would try every possible values. However, we can save significant amount of time training if we know the optimal size of the bottleneck in advance.

- baseline: Trying every possible value, from 1 to 3.
- **CubeDimAE**: Dimension estimation, followed by training the autoencoder *only* *once*.

<div align="center">

### Baseline

Dataset | AE1 | AE2 | AE3 | Total (*s*)
---: | ---: | ---: | ---: | ---:
S curve | 7.29 | 7.25 | 7.66 | 22.20
Swiss roll | 6.92 | 7.06 | 7.34 | 21.32
Möbius strip | 6.94 | 7.08 | 7.32 | 21.34
Hollow sphere | 7.06 | 7.07 | 7.35 | 21.48
Solid sphere | 6.97 | 7.09 | 7.33 | 21.39

### CubeDimAE (≈40% saved)

Dataset | Estimation | AE | Total (*s*)
---: | ---: | ---: | ---:
S curve | 3.20 | 7.25 | 10.45
Swiss roll | 5.42 | 7.06 | 12.48
Möbius strip | 3.03 | 7.08 | 10.11
Hollow sphere | 5.67 | 7.07 | 12.74
Solid sphere | 10.12 | 7.33 | 17.45

</div>

## Reproduction

Experiments were run on a Macbook M2 pro.

> [!NOTE]
> *The scripts were written solely for reproducing the experimental results. They were not designed for reuse.*

### Environment

Platform: 'darwin'<br>
Package Manager: Conda<br>
Packages:<br>
```yaml
name: CubeDimAE
channels:
  - defaults
dependencies:
  - python=3.11.0
  - numpy=1.23.5
  - matplotlib=3.9.2
  - tensorflow=2.12.0
  - tqdm=4.66.5
  - pyyaml=6.0.2
  - scikit-learn=1.5.2
```
Later versions and different platforms will likely have no issue.

### Run

```bash
conda update conda
conda env create --file environment.yml
conda activate CubeDimAE

python experiment.py --config config.yml
```

## Citation

```bibtex
@inproceedings{kim2025cubedimae,
  title={CubeDimAE: Automatic Autoencoder Generation based on Dimension Estimation by Tessellation},
  author={Kim, Byungrok and Hwang, Myeong-Ha and Joo, Jeonghyun and Kwon, YooJin and Lee, Hyunwoo},
  booktitle={2025 IEEE International Conference on Big Data and Smart Computing (BigComp)},
  pages={20--25},
  year={2025},
  organization={IEEE}
}
```
