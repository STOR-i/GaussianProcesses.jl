# Linear mean function

@doc """
# Description
Constructor for the Linear mean function

m(x) = xᵀβ
# Arguments:
* `β::Vector{Float64}`: One coefficient for each dimension
""" ->
type MeanLin <: Mean
    β::Vector{Float64}
    dim::Int
    priors::Array          # Array of priors for mean parameters
    MeanLin(β::Vector{Float64}) = new(β, length(β),[])
end

function mean(mLin::MeanLin,x::MatF64)
    dim, nobsv = size(x)
    dim == mLin.dim || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    z = zeros(nobsv)
    for i in 1:nobsv
        z[i] = dot(x[:, i], mLin.β)
    end
    return z
end    

get_params(mLin::MeanLin) = mLin.β
get_param_names(::MeanLin) = [:β]
num_params(mLin::MeanLin) = mLin.dim

function set_params!(mLin::MeanLin, hyp::Vector{Float64})
    length(hyp) == mLin.dim || throw(ArgumentError("Linear mean function only has $(mLin.dim) parameters"))
    mLin.β = hyp
end

function grad_mean(mLin::MeanLin, x::Vector{Float64})
    dM_theta = x
    return dM_theta
end
