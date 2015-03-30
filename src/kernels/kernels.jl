#This file contains a list of the currently available covariance functions

import Distributions.params
import Base.show

abstract Kernel

function show(io::IO, k::Kernel, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(k)), Params: ")
    show(io, params(k))
    print(io, "\n")
end
    
# Isotropic kernels
include("se.jl")                # Squared exponential covariance function
include("mat.jl")               # Matern covariance function
include("peri.jl")              # Periodic covariance function
include("poly.jl")              # Polnomial covariance function
include("rq.jl")                # Rational quadratic covariance function
include("lin.jl")               # Linear covariance function

# ARD kernels
include("se_ard.jl")            # Squared exponential
include("rq_ard.jl")            # Rational quadratic
include("lin_ard.jl")           # Linear covariance
include("mat_ard.jl")           # Matern covariance

# Composite kernels
include("sum_kernel.jl")        # Sum of kernels
include("prod_kernel.jl")       # Product of kernels
