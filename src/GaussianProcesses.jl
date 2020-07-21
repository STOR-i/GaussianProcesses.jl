module GaussianProcesses

using Distributions, LinearAlgebra, Distances
import Statistics: mean, cov

export GP, GPR, SquaredExponential, Stationary, KernelParameter, Zero, cov, marginal_log_likelihood, return_params, get_objective

include("parameters.jl")
include("kernels/kernels.jl")
include("mean_functions.jl")
include("GP.jl")
include("inference/Inference.jl")

end
