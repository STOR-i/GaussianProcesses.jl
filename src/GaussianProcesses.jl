module GaussianProcesses

using Distributions, LinearAlgebra
import Statistics: mean, cov

export GP, GPR, SquaredExponential, Stationary, KernelParameter, Zero, cov, marginal_log_likelihood, return_params 

include("parameters.jl")
include("kernels/kernels.jl")
include("mean_functions.jl")
include("GP.jl")
include("inference/Inference.jl")

end
