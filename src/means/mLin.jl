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
    priors::Array      # Array of priors for mean parameters
    MeanLin(β::Vector{Float64}) = new(β, [])
end

mean(mLin::MeanLin, x::VecF64) = dot(mLin.β, x)
mean(mLin::MeanLin, X::MatF64) = X'mLin.β

get_params(mLin::MeanLin) = mLin.β
get_param_names(::MeanLin) = [:β]
num_params(mLin::MeanLin) = length(mLin.β)

function set_params!(mLin::MeanLin, hyp::Vector{Float64})
    length(hyp) == length(mLin.β) || throw(ArgumentError("Linear mean function only has $(mLin.dim) parameters"))
    mLin.β = hyp
end

function grad_mean(mLin::MeanLin, x::Vector{Float64})
    dM_theta = x
    return dM_theta
end
