module GaussianProcesses
using Optim, PDMats, Distances, FastGaussQuadrature, Compat, RecipesBase, Distributions
import Compat: view, cholfact!
import Distributions: logpdf, gradlogpdf
import Base: +, *, rand, rand!, mean, cov, push!

# Functions that should be available to package users should be explicitly exported here

export GPBase, GP, GPE, GPMC, GPV, predict_f, predict_y, Kernel, Likelihood, CompositeKernel, SumKernel, ProdKernel, Masked, FixedKern, fix, Noise, Const, SE, SEIso, SEArd, Periodic, Poly, RQ, RQIso, RQArd, Lin, LinIso, LinArd, Matern, Mat12Iso, Mat12Ard, Mat32Iso, Mat32Ard, Mat52Iso, Mat52Ard, # Kernel functions
    MeanZero, MeanConst, MeanLin, MeanPoly, SumMean, ProdMean, # Mean functions
    GaussLik, BernLik, ExpLik, StuTLik, PoisLik, BinLik,       # Likelihood functions
    mcmc, optimize!,                                           # Inference functions
    set_priors!,set_params!, update_target!                                                


const MatF64 = AbstractMatrix{Float64}
const VecF64 = AbstractVector{Float64}

# all package code should be included here
include("means/means.jl")              # Mean functions
include("kernels/kernels.jl")          # Kernel functions
include("likelihoods/likelihoods.jl")  # Likelihood functions
include("utils.jl")                    # Additional utility functions
include("chol_utils.jl")               # Cholesky functions
include("GP.jl")                       # GP base
include("GPE.jl")                      # Exact GP for regression
include("GPMC.jl")                     # Monte Carlo GP
include("GPV.jl")                      # Variational GP
include("mcmc.jl")                     # MCMC sampler
include("optimize.jl")                 # Optim.jl interface
include("plot.jl")                     # Plotting functions

# ScikitLearnBase, which is a skeleton package.
include("ScikitLearn.jl")

end # module
