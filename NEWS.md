# News

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
