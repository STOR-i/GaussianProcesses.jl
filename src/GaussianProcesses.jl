module GaussianProcesses

using Optim, PDMats, ElasticPDMats, ElasticArrays, Distances, FastGaussQuadrature, RecipesBase, Distributions, Zygote, ProgressBars, Plots
using StaticArrays
using StatsFuns, SpecialFunctions

using LinearAlgebra, Printf, Random, Statistics
import Statistics: mean, cov, var
import Base: size, push!
import PDMats: dim, Matrix, diag, pdadd!, *, \, inv, logdet, eigmax, eigmin, whiten!, unwhiten!, quad, quad!, invquad, invquad!, X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X
import RecipesBase: plot
import Plots: plot
# Functions that should be available to package
# users should be explicitly exported here
export GPBase, GP, GPE, GPA, ElasticGPE, ClfMetric, Chain, Approx, predict_f, predict_y, Kernel, Likelihood, CompositeKernel, SumKernel, ProdKernel, Masked, FixedKernel, fix, Noise, Const, SE, SEIso, SEArd, Periodic, Poly, RQ, RQIso, RQArd, Lin, LinIso, LinArd, Matern, Mat12Iso, Mat12Ard, Mat32Iso, Mat32Ard, Mat52Iso, Mat52Ard, #kernel functions
    MeanZero, MeanConst, MeanLin, MeanPoly, SumMean, ProdMean, MeanPeriodic, #mean functions
    GaussLik, BernLik, ExpLik, StuTLik, PoisLik, BinLik,       #likelihood functions
    mcmc, svgd, ess, lss, optimize!, vi, var_exp, dv_var_exp, elbo, initialise_Q,    #inference functions
    set_priors!,set_params!, update_target!, autodiff, update_Q!, evaluate, proba, accuracy, precision, recall, rmse, mode, classification, plot, chain, push!, plot
using ForwardDiff: GradientConfig, Dual, partials, copyto!, Chunk
import ForwardDiff: seed!


const φ = normpdf
const Φ = normcdf
const invΦ = norminvcdf

# all package code should be included here
include("means/means.jl")
include("kernels/kernels.jl")
include("likelihoods/likelihoods.jl")
include("common.jl")
include("chol_utils.jl")
include("GP.jl")
include("GPE.jl")
include("GPEelastic.jl")
include("GPA.jl")
include("utils.jl")
include("vi.jl")
include("mcmc.jl")
include("svgd.jl")
include("optimize.jl")
include("crossvalidation.jl")
include("plot.jl")
include("sparse/sparseGP.jl")
include("metrics.jl")


# ScikitLearnBase, which is a skeleton package.
include("ScikitLearn.jl")

end # module
