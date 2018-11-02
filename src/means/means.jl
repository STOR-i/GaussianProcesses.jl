#This file contains a list of the currently available mean functions

abstract type Mean end

# Calculate mean for matrix of observations
function mean(m::Mean, X::AbstractMatrix)
    nobs = size(X, 2)
    μ = Array{Float64}(undef, nobs)
    @inbounds for i in 1:nobs
        μ[i] = mean(m, view(X, :, i))
    end
    μ
end

# Calculates the stack [dm / dθᵢ] of mean matrix gradients
function grad_stack(m::Mean, X::AbstractMatrix)
    nobs = size(X, 2)
    mat = Array{Float64}(undef, nobs, num_params(m))
    @inbounds for i in 1:nobs
        mat[i, :] = grad_mean(m, view(X, :, i))
    end
    mat
end

#————————————————————————————————————————————

include("mZero.jl")          # Zero mean function
include("mConst.jl")         # Constant mean function
include("mLin.jl")           # Linear mean function
include("mPoly.jl")          # Polynomial mean function
include("composite_mean.jl") # Composite means
