# Polynomial mean function

@doc """
# Description
Constructor for the Polynomial mean function

m(x) = Σᵢⱼ βᵢⱼ xᵢʲ
# Arguments:
* `β::Matrix{Float64}`: A d x D matrix of coefficients where d is the dimension of the observations and D is the degree of the polynomial
""" ->
type MeanPoly <: Mean
    β::Matrix{Float64}    # Polynomial coefficients
    priors::Array         # Array of priors for mean parameters
    MeanPoly(β::Matrix{Float64}) = new(β, [])
end

function mean(mPoly::MeanPoly, x::VecF64)
    dim = length(x)
    deg = size(mPoly.β, 2)
    dim == size(mPoly.β, 1) || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    return sum(dot(x.^j, mPoly.β[:,j]) for j in 1:deg)
end


get_params(mPoly::MeanPoly) = vec(mPoly.β)
get_param_names(mPoly::MeanPoly) = [:β]
num_params(mPoly::MeanPoly) = length(mPoly.β)

function set_params!(mPoly::MeanPoly, hyp::Vector{Float64})
    length(hyp) == num_params(mPoly) || throw(ArgumentError("Polynomial mean function has $(num_param) parameters"))
    mPoly.β = reshape(hyp, size(mPoly.β)...)
end


function grad_mean(mPoly::MeanPoly, x::VecF64)
    dim = length(x)
    deg = size(mPoly.β, 2)
    dM_theta = Array{Float64}(dim, deg)
    for i in 1:dim
        for j in 1:deg
            dM_theta[i,j] = x[i].^j
        end
    end
    return vec(dM_theta)
end
