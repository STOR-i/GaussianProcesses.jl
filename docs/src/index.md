# GaussianProcesses

## Introduction

Gaussian processes are a family of stochastic processes which provide a flexible nonparametric tool for modelling data.
A Gaussian Process places a prior over functions, and can be described as an infinite dimensional generalisation of a multivariate Normal distribution.
Moreover, the joint distribution of any finite collection of points is a multivariate Normal.
This process can be fully characterised by its mean and covariance functions, where the mean of any point in the process is described by the *mean function* and the covariance between any two observations is specified by the *kernel*.
Given a set of observed real-valued points over a space, the Gaussian Process is used to make inference on the values at the remaining points in the space.

For an extensive review of Gaussian Processes there is an excellent book [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) by Rasmussen and Williams, (2006)

## Installation

GaussianProcesses.jl requires Julia version 1.0 or above.
To install GaussianProcesses.jl run the following command inside the Julia package REPL

```julia
pkg> add GaussianProcesses
```

or in the standard REPL

```julia
julia> using Pkg
julia> Pkg.add("GaussianProcesses")
```

## Supported features

1. Inference methods
    * Exact methods based on linear algebra for Gaussian processes with a normal likelihood;
    * Hamiltonian Monte Carlo for Gaussian processes with any other likelihood;
    * Elliptical slice sampler for Gaussian processes with a Gaussian likelihood;
    * Variational inference for Gaussian processes with any likelihood;
    * Sparse approximations to accelerate inference through pseudo-inputs;
2. Hyperparameters
    * Optimization of the marginal likelihood for exact GPs;
    * Posterior samples of the hyperparameters for likelihoods other than normal;
    * Methods to obtain the cross-validation score and its derivative are also available;
3. Kernels
    * Basic kernels such as squared exponential (aka radial basis function), Matérn, etc.;
    * Sum and product kernels;
    * Masked kernels, to apply a kernel to a subset of input dimensions;
    * Fixed kernels, to prevent optimization of certain kernel hyperparameters;
    * Autodifferentation of user-implemented kernels (work in progress);
4. Mean functions
    * Basic mean functions, such as constant, linear, periodic, etc.;
    * The parameters of the mean functions can be fitted by maximum likelihood for exact GPs;
5. Easy access to Gaussian process methods
    * Underlying methods like the covariances and their derivatives,

## Features currently not implemented

There are many features that we would love to add to this package.
If you need one of these features and would be interested in contributing to the package,
please get in touch or submit a pull request through GitHub.

* Tuning of Hamiltonian Monte Carlo for efficient posterior draws;
* Multivariate Gaussian processes, when the output is a vector, which encompasses
  multi-task GPs, cokriging, multiclass classification.
* Approximation methods for large GPs:
    * expectation propagation;
    * Laplace approximations;
