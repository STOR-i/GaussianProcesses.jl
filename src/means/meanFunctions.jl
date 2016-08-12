#This file contains a list of the currently available mean functions

import Base.show

abstract Mean

function show(io::IO, m::Mean, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(m)), Params: ")
    show(io, get_params(m))
    print(io, "\n")
end

# Calculates the stack [dm / dθᵢ] of mean matrix gradients
function grad_stack(m::Mean, X::Matrix{Float64})
    n = num_params(m)
    d, nobsv = size(X)
    mat = Array(Float64, nobsv, n)
    for i in 1:nobsv
        @inbounds mat[i,:] = grad_mean(m, X[:,i])
    end
    return mat
end

include("mConst.jl")         # Constant mean function, which also contains the zero mean function
include("mLin.jl")           # Linear mean function
include("mPoly.jl")          # Polynomial mean function
include("sum_mean.jl")       # Sum mean functions
include("prod_mean.jl")      # Product of mean functions

