module GaussianProcesses

using Optim, PDMats, Distances, FastGaussQuadrature, RecipesBase, Distributions
using StatsFuns, SpecialFunctions

using LinearAlgebra, Printf, Random, Statistics

# Functions that should be available to package
# users should be explicitly exported here
    MeanZero, MeanConst, MeanLin, MeanPoly, SumMean, ProdMean, #mean functions
    GaussLik, BernLik, ExpLik, StuTLik, PoisLik, BinLik,       #likelihood functions
    mcmc, optimize!,                                           #inference functions
    set_priors!,set_params!, update_target!


const MatF64 = AbstractMatrix{Float64}
const VecF64 = AbstractVector{Float64}

const φ = normpdf
const Φ = normcdf
const invΦ = norminvcdf

# all package code should be included here
include("means/means.jl")
include("kernels/kernels.jl")
include("likelihoods/likelihoods.jl")
include("common.jl")
include("utils.jl")
include("chol_utils.jl")
include("GP.jl")
include("GPE.jl")
include("GPMC.jl")
include("mcmc.jl")
include("optimize.jl")
include("plot.jl")

# ScikitLearnBase, which is a skeleton package.
include("ScikitLearn.jl")

end # module
