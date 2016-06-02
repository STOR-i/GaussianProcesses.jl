# News

## Version 0.1.5 (2016-06-02)
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
