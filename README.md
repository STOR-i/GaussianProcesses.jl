# GaussianProcesses.jl

A Gaussian Processes package for Julia. 

This package is still under development. If you have any suggestions to improve the package, or if you've noticed a bug, then please post an [issue](https://github.com/STOR-i/GaussianProcesses.jl/issues/new) for us and we'll get to it as quickly as we can. Pull requests are also welcome.

## Introduction

Gaussian processes are a family of stochastic processes which provide a flexible nonparametric tool for modelling data. A Gaussian Process places a prior over functions, and can be described as an infinite dimensional generalisation of a multivariate Normal distribution. Moreover, the joint distribution of any finite collection of points is a multivariate Normal. This process can be fully characterised by its mean and covariance functions, where the mean of any point in the process is described by the *mean function* and the covariance between any two observations is specified by the *kernel*. Given a set of observed real-valued points over a space, the Gaussian Process is used to make inference on the values at the remaining points in the space.

For an extensive review of Gaussian Processes there is an excellent book [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) by Rasmussen and Williams, (2006)

## Installation

GaussianProcesses.jl requires Julia version 0.5 or above. To install GaussianProcesses.jl run the following command inside a Julia session:

```julia
julia> Pkg.add("GaussianProcesses")
```
## Functionality

The package allows the user to fit exact **Gaussian process** models when the observations are Gaussian distributed about the latent function. In the case where the *observations are non-Gaussian*, the posterior distribution of the latent function is intractable. The package allows for Monte Carlo sampling from the posterior.

The main function of the package is `GP`, which fits the Gaussian process
```
gp = GP(X',y,mean,kernel)
gp = GP(X',y,mean,kernel,likelihood)
```
for Gaussian and non-Gaussian data respectively.

The package has a number of *mean*, *kernel* and *likelihood* functions available. See the documentation for further details.

### Inference

The parameters of the model can be estimated by maximizing the log-likelihood (where the latent function is integrated out) using the `optimize!` function, or in the case of *non-Gaussian data*, an `mcmc` function is available, utilizing the Hamiltonian Monte Carlo sampler, and can be used to infer the model parameters and latent function values. 
```
optimize!(gp)    # Find parameters which maximize the log-likelihood
mcmc(gp)         # Sample from the GP posterior
```
See the [notebooks](https://github.com/STOR-i/GaussianProcesses.jl/tree/master/notebooks) for examples of the functions used in the package.

## Documentation

Documentation is accessible in the Julia REPL in help mode. Help mode can be started by typing '?' at the prompt.

```julia
julia> ?
help?> GP

[type]

GaussianProcesses.GP

                                  Description
                                 -–––––––––––-

  Fits a Gaussian process to a set of training points. The Gaussian process is
  defined in terms of its mean and covariance (kernel) functions, which are
  user defined. As a default it is assumed that the observations are noise
  free.

                                   Arguments:
                                  -––––––––––-

    • `X::Matrix{Float64}`: Training inputs
    • `y::Vector{Float64}`: Observations
    • `m::Mean`           : Mean function
    • `k::kernel`         : Covariance function
    • `logNoise::Float64` : Log of the observation noise. The default is
      -1e8, which is equivalent to assuming no observation noise.

                                    Returns:
                                   -––––––––-

    • `gp::GP`            : A Gaussian process fitted to the training data

 Details:

	source: (16,"/home/jamie/.julia/v0.4/GaussianProcesses/src/GP.jl")
```

## Notebooks

Sample code is available from the [notebooks](https://github.com/STOR-i/GaussianProcesses.jl/tree/master/notebooks)

## Related packages

[GeoStats](https://github.com/juliohm/GeoStats.jl) - High-performance implementations of geostatistical algorithms for the Julia programming language. This package is in its initial development, and currently only contains Kriging estimation methods. More features will be added as the Julia type system matures.

## ScikitLearn

This package also supports the [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl) interface. ScikitLearn provides many tools for machine learning such as hyperparameter tuning and cross-validation. See [here](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Gaussian_Processes_Julia.ipynb) for an example of its usage with this package.
