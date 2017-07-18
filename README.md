# GaussianProcesses.jl

A Gaussian Processes package for Julia. 

This package is still in the early stages of development. If you have any suggestions to improve the package, or if you've noticed a bug, then please post an [issue](https://github.com/STOR-i/GaussianProcesses.jl/issues/new) for us and we'll get to it as quickly as we can. Pull requests are also welcome.

## Introduction

Gaussian processes are a family of stochastic processes which provide a flexible nonparametric tool for modelling data. A Gaussian Process places a prior over functions, and can be described as an infinite dimensional generalisation of a multivariate Normal distribution. Moreover, the joint distribution of any finite collection of points is a multivariate Normal. This process can be fully characterised by its mean and covariance functions, where the mean of any point in the process is described by the *mean function* and the covariance between any two observations is specified by the *kernel*. Given a set of observed real-valued points over a space, the Gaussian Process is used to make inference on the values at the remaining points in the space.

For an extensive review of Gaussian Processes there is an excellent book [Gaussian Processes for Machine Learning] (http://www.gaussianprocess.org/gpml/chapters/RW.pdf) by Rasmussen and Williams, (2006).

## Installation

GaussianProcesses requires Julia version 0.4 or above. To install GaussianProcesses run the following command inside a Julia session:

```julia
julia> Pkg.add("GaussianProcesses")
```

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



## 1-dimensional regression example

The first step in modelling with Gaussian Processes is to choose mean functions and kernels which describe the process. GaussianProcesses can be optionally used with a plotting package. Currently the packages [Gadfly] (https://github.com/dcjones/Gadfly.jl) and [PyPlot] (https://github.com/stevengj/PyPlot.jl) are supported.

```julia
using PyPlot, GaussianProcesses

# Training data
n = 10
x = 2π * rand(n)              
y = sin(x) + 0.05*randn(n)

# Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = SE(0.0,0.0)                   # Squared exponential kernel with parameters
                                     # log(ℓ) = 0.0, log(σ) = 0.0
```

Note that the parameters of the kernel are given on the log-scale. This is true
for all strictly positive hyperparameters. Gaussian Processes are represented
by objects of type `GP` and constructed from observation data, a mean function and kernel, and optionally the amount of observation noise. 

```julia
logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
gp = GP(x,y,mZero,kern, logObsNoise)      # Fit the GP

  Dim = 1
  Number of observations = 10
  Mean function:
    Type: MeanConst, Params: [0.0]
  Kernel:
    Type: SEIso, Params: [0.0,0.0]
  Input observations = 
1x10 Array{Float64,2}:
 2.20808  3.62094  4.10836  2.76144  …  5.60978  4.22383  0.289366  5.61989
  Output observations = [0.863383,-0.578097,-0.746813,0.339261,0.279411,0.35201,-0.565207,-0.947441,0.317193,-0.68688]
  Variance of observation noise = 0.36787944117144233
  Marginal Log-Likelihood = -9.056
```

Once we've fit the GP function to the data, we can calculate the predicted mean and variance of of the function at unobserved points. This is done with the `predict` function.

```
μ, σ² = predict(gp,linspace(0,2π,100))
```

The predict function is implicitly used when plotting the GP. Plotting is straightforward to apply, but the display will depend on the package loaded at the start of the session (e.g. PyPlot or Gadfly). Note that, at present, the plotting package should be loaded before `GaussianProcesses`. The plot function outputs the predicted mean (blue line) and the uncertainty in the function is given by the confidence bands, which are set to 95% by default. 
```julia
plot(gp)
```
![1-D Gaussian Process](/docs/regression_1d.png "1-D Gaussian Process pre-optimization")

The hyperparameters are optimized using the [Optim](https://github.com/JuliaOpt/Optim.jl) package. This offers users a range of optimization algorithms which can be applied to estimate the hyperparameters using type II maximum likelihood estimation. Gradients are available for all mean and kernel functions used in the package and therefore it is recommended that the user utilizes gradient based optimization techniques. As a default, the `optimize!` function uses the `Conjugate Gradients` solver, however, alternative solvers can be applied. 
```julia
optimize!(gp)   #Optimise the hyperparameters
plot(gp)       #Plot the GP after the hyperparameters have been optimised 
```

![1-D Gaussian Process](/docs/regression_1da.png?raw=true "1-D Gaussian Process post-optimization")

## 2-dimensional regression example

This is a simple 2-D regression example. 
```julia
using Gadfly, GaussianProcesses

#Training data
d, n = 2, 50         # Dimension and number of observations

x = 2π * rand(d, n)                               
y = vec(sin(x[1,:]).*sin(x[2,:])) + 0.05*rand(n) 
```
For problems of dimension>1 we can use isotropic (Iso) kernels or automatic relevance determination (ARD) kernels. These are implemented automatically by the user based on the choice of hyperparameters. For example, below we use the Matern 5/2 ARD kernel, if we wanted to use the Iso alternative then we would set the kernel as `kern=Mat(5/2,0.0,0.0)`.

In this example we use a composite kernel represented as the sum of a Matern 5/2 ARD kernel and a Squared Exponential Iso kernel. This is easily implemented using the `+` symbol, or in the case of a product kernel, using the `*` symbol (i.e. `kern = Mat(5/2,[0.0,0.0],0.0) \* SE(0.0,0.0)`).
```julia
#Choose mean and covariance function
mZero = MeanZero()                             # Zero mean function
kern = Mat(5/2,[0.0,0.0],0.0) + SE(0.0,0.0)    # Sum kernel with Matern 5/2 ARD kernel 
                                               # with parameters [log(ℓ₁), log(ℓ₂)] = [0,0] and log(σ) = 0
                                               # and Squared Exponential Iso kernel with
                                               # parameters log(ℓ) = 0 and log(σ) = 0
```
Fit the Gaussian process to the data using the prespecfied mean and covariance functions.
```julia
logObsNoise = -2.0                    # log standard deviation of observation noise (this is optional)
gp = GP(x,y,mZero,kern,logObsNoise)   # Fit the GP
```

Using the [Optim](https://github.com/JuliaOpt/Optim.jl) package we have the option to choose from a range of optimize functions including conjugate gradients. It is also possible to fix the hyperparameters in either the mean, kernel or observation noise, by settting them to false in `optimize!` (e.g. `optimize!(...,mean=false)`).

```julia
optimize!(gp; method=Optim.ConjugateGradient())                   # Optimize the hyperparameters

Results of Optimization Algorithm
 * Algorithm: Conjugate Gradient
 * Starting Point: [-2.0,0.0,0.0,0.0,0.0,0.0,0.0]
 * Minimum: [-2.4508978539051807,-0.47461801552770505,0.43590368883158864,0.3401816678461438,-0.768005238815202,0.0788474114802276,-1.2824416535170349]
 * Value of Function at Minimum: 15.307393
 * Iterations: 3
 * Convergence: true
   * |x - x'| < 1.0e-32: false
   * |f(x) - f(x')| / |f(x)| < 1.0e-08: true
   * |g(x)| < 1.0e-08: false
   * Exceeded Maximum Number of Iterations: false
 * Objective Function Calls: 83
 * Gradient Call: 80
```
Notice that the syntax for plotting the GP is the same for Gadfly as for PyPlot.
```julia
plot(gp; clim=(-10.0, 10.0,-10.0,10.0)) # Plot the GP over range clim
```

![2-D Gaussian Process](/docs/regression_2d.png?raw=true "2-D Gaussian Process")

## Sampling from the GP

After specifying a mean and covariance function it's straightforward to sample from the GP prior using the `rand` function.

```
# Select mean and covariance function
mZero = MeanZero()
kern = SE(0.0,0.0)*Mat(3/2,2.0,0.5) 

# Specify the GP prior
gp = GP(m=mZero,k=kern)     

x_path = collect(linspace(-5,5)) #Range to sample over

prior=rand(gp,x_path, 10)  
```

![Gaussian Process Prior](/docs/prior_samples.png?raw=true "Gaussian Process Prior")


Once we have some data we can then update the GP to give the posterior distribution, and again using the `rand` function, we can sample from the GP.

```
# Training data
x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

# Fit data to GP object
GaussianProcesses.fit!(gp, x, y)
post=rand(gp,x_path, 10)
```
![Gaussian Process Prior](/docs/posterior_samples.png?raw=true "Gaussian Process Posterior")

## ScikitLearn

This package also supports the [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl) interface. ScikitLearn provides many tools for machine learning such as hyperparameter tuning and cross-validation. See [here](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Gaussian_Processes_Julia.ipynb) for an example of its usage with this package.
