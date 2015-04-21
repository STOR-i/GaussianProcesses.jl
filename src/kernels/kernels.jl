#This file contains a list of the currently available covariance functions

import Base.show

abstract Kernel

function show(io::IO, k::Kernel, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(k)), Params: ")
    show(io, get_params(k))
    print(io, "\n")
end

include("lin.jl")               # Linear covariance function
include("se.jl")                # Squared exponential covariance function
include("rq.jl")                # Rational quadratic covariance function
include("mat.jl")                # Matern covariance function
include("peri.jl")              # Periodic covariance function
include("poly.jl")              # Polnomial covariance function

# Composite kernels
include("sum_kernel.jl")        # Sum of kernels
include("prod_kernel.jl")       # Product of kernels
