# GaP.jl

A Gaussian Processes package for Julia.

## Introduction

Gaussian processes are a family of stochastic processes which provide a flexible tool for modelling data. A Gaussian Process takes place in a finite dimensional real space, and to each point in that space is a corresponding Normal random variable. Moreover, the joint distribution of any finite collections of points is multivariate Normal. The mean of any point in the process is described by the *mean function* of the process and the covariance
between any two points by the process *kernel*. Given a set observed real-values for points over a space, the Gaussian Process is used to make inference on the values at the remaining points in the space.

## Basic Usage

The first step in modelling with Gaussian Processes is to choose mean functions and kernels which describe the process. GaP can be optionally used with a plotting package. Currently the packages Gadfly and Winston are supported.

```julia
using Winston, GaP

# Training data
n = 10
x = 2π * rand(n)              
y = sin(x) + 0.05*randn(n)

# Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = SE(0.0,0.0)                   # Squared exponential kernel with parameters
                                     # log(l) = 0.0, log(σ) = 0.0
```

Note that the parameters of the kernel are given on the log-scale. This is true
for all strictly positive hyperparameters. Gaussian Processes are represented
by objects of type `GP` and constructed from observation data, a mean function and kernel, and optionally the amount of observation noise. If one of the plotting packages has been loaded (prior to GaP) then the function `plot` can be used to visualise the fitted process.

```julia
logObsNoise = -1.0                        # log standard deviation of observation noise
gp = GP(x,y,mZero,kern, logObsNoise)      # Fit the GP
plot(gp)
```
![1-D Gaussian Process](/docs/regression_1d.png?raw=true "1-D Gaussian Process")

![1-D Gaussian Process](/docs/regression_1da.png?raw=true "1-D Gaussian Process")
