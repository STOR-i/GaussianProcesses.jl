# News

## Version 0.9.0 (2018-12-07)
* Introduction of `ElasticGPE` to allow `GP` which can grow without
  "refitting" whole Gaussian process (see also [#88](https://github.com/STOR-i/GaussianProcesses.jl/issues/88)).
* Various performance and interface improvements to Kernels
* Introduction of `ADKernel` to simplify introduction of new/custom kernels
  through use of auto-differentiation.

## Version 0.8.0 (2018-10-02)
* Updated Julia requirement to v1.0
* Updated requirement of `RecipesBase.jl` to v0.6

Note: Could not create release compatible with both Julia v0.7 and Julia v1.0 due to `RecipesBase.jl` dependency

## Version 0.7.0 (2018-09-28)
* Updated Julia requirement to v0.7
* Performance improvements to Kernels
* Added dependencies on `StatsFuns` and `SpecialFunctions`
* Removed dependency on `Compat`
* Renamed `FixedKern` to `FixedKernel`
* Added type parameters to `GPE`, `GPMC`, `ProdKernel`, and `SumKernel`, `ProdMean` and `SumMean`
* Renamed fields of `GPE` and `GPE` (`x` instead of `X`, `mean` instead of `m`, `kernel` instead of `k`, and `nobs` instead of `nobsv`)
* Renamed fields of `FixedKernel` and `Masked` (kernel instead of kern)
* Renamed fields of `ProdKernel` and `SumKernel` (kernels instead of kerns)
* Renamed keyword arguments of the `GPE` constructor to kernel and mean
* Renamed function `subkernels` and `submeans` to `components`


## Version 0.6.0 (2018-03-02)
* Updated optimization code to be compatible with new Optim.jl API
* Removed Klara dependency
* Performance improvements to `predict` functions


## Version 0.5.0 (2017-10-18)
* Updated Julia version requirement to 0.6
* GP type has been renamed to GPE (GP exact) for Gaussian likelihoods
* Introduced GPMC type for fitting models with non-Gaussian likelihoods:
  * Bernouilli, Poisson, Binomial and student-t likelihoods available
* Introduced priors for parameters of the kernal, mean, and likelihood functions
* MCMC available for GPE and GPMC type
* Changed plotting functions to use [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
* Created notebooks illustrating package features


## Version 0.4.0 (2016-10-04)
* Julia requirement moved up to version 0.5
* Major speed improvements for fitting of GP object, and for covariance and gradient calculations
* New `Masked` kernel
* Various bug fixes

## Version 0.3.0 (2016-07-11)
* Introduced `KernelData` type to recycle calculations
* Removed Winston plotting functions and implemented PyPlot as an alternative
* Created methods for `mean` and `cov` functions of the `Mean` and `Kernel` objects
* Fixed `optimize!` function to be consistent with most recent version of Optim.jl 
* Improvements to the `Periodic` kernel
* `fit!` function no longer exported due to clash with a few packages

## Version 0.2.1 (2016-06-06)
* Added fit! function to fit a new set observations to existing GP object

## Version 0.2.0 (2016-06-03)
* Julia requirement moved up to v0.4
* Support added for ScikitLearn
* rand and rand! functions added to sample prior and posterior paths of Gaussian process
* Major speed improvements for gradient calculations of stationary ARD kernels
* Minor fixes for some kernels

## Version 0.1.4 (2015-10-28)
* Fixed plotting deprecation errors with Julia 0.4

## Version 0.1.3 (2015-10-26)

* Major speed improvements to kernel calculations, in particular to stationary and composite kernels
* Fixed depraction warnings for Julia v0.4
* All stationary kernels have the super type Stationary
* Distance matrix calculations outsourced to Distances

## Version 0.1.2 (2015-06-04)

* Improvements in speed for predict and fitting functions
* Positive definite matrix calculations outsourced to PDMats
