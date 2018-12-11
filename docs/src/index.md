# GaussianProcesses

## Introduction

Gaussian processes are a family of stochastic processes which provide a flexible nonparametric tool for modelling data.
A Gaussian Process places a prior over functions, and can be described as an infinite dimensional generalisation of a multivariate Normal distribution.
Moreover, the joint distribution of any finite collection of points is a multivariate Normal.
This process can be fully characterised by its mean and covariance functions, where the mean of any point in the process is described by the *mean function* and the covariance between any two observations is specified by the *kernel*.
Given a set of observed real-valued points over a space, the Gaussian Process is used to make inference on the values at the remaining points in the space.

For an extensive review of Gaussian Processes there is an excellent book [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) by Rasmussen and Williams, (2006)

## Installation

GaussianProcesses.jl requires Julia version 0.5 or above.
To install GaussianProcesses.jl run the following command inside the Julia package REPL

```julia
pkg> add GaussianProcesses
```

or in the standard REPL

```julia
julia> using Pkg
julia> Pkg.add("GaussianProcesses")
```
