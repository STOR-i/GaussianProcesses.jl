module GaussianProcesses
using Optim, PDMats, Distances, FastGaussQuadrature
using Compat
using RecipesBase
import Compat: view, cholfact!
using Distributions
import Distributions: logpdf, gradlogpdf
import Base: +, *
import Base: rand, rand!, mean, cov, push!

# Functions that should be available to package
# users should be explicitly exported here

export GPBase, GP, GPE, GPMC, predict_f, predict_y, Kernel, Likelihood, CompositeKernel, SumKernel, ProdKernel, Masked, FixedKern, fix, Noise, Const, SE, SEIso, SEArd, Periodic, Poly, RQ, RQIso, RQArd, Lin, LinIso, LinArd, Matern, Mat12Iso, Mat12Ard, Mat32Iso, Mat32Ard, Mat52Iso, Mat52Ard, #kernel functions
    MeanZero, MeanConst, MeanLin, MeanPoly, SumMean, ProdMean, #mean functions
    GaussLik, BernLik, ExpLik, StuTLik, PoisLik, BinLik,       #likelihood functions
    mcmc, optimize!,                                           #inference functions
    set_priors!,set_params!, update_target!                                                


const MatF64 = AbstractMatrix{Float64}
const VecF64 = AbstractVector{Float64}

# all package code should be included here
include("means/means.jl")
include("kernels/kernels.jl")
include("likelihoods/likelihoods.jl")
include("utils.jl")
include("chol_utils.jl")
include("GP.jl")
include("GPE.jl")
include("GPMC.jl")
# include("mcmc.jl")
include("optimize.jl")
include("plot.jl")

# ScikitLearnBase, which is a skeleton package.
include("ScikitLearn.jl")

end # module
