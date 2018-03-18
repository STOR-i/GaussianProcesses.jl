#Constant mean function

@doc """
# Description
Constructor for the constant mean function

m(x) = β
# Arguments:
* `β::Float64`: Constant
""" ->
type MeanConst <: Mean
    β::Float64
    priors::Array          # Array of priors for mean parameters
    MeanConst(β::Float64) = new(β, [])
end

mean(mConst::MeanConst, x::VecF64) = mConst.β
mean(mConst::MeanConst, X::MatF64) = fill(mConst.β, size(X,2))

get_params(mConst::MeanConst) = Float64[mConst.β]
get_param_names(::MeanConst) = [:β]
num_params(mConst::MeanConst) = 1
function set_params!(mConst::MeanConst, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Constant mean function only has 1 parameter"))
    mConst.β = hyp[1]
end
function grad_mean(mConst::MeanConst, x::VecF64)
    dM_theta = ones(1)
    return dM_theta
end
