#Constant mean function

@doc """
# Description
Constructor for the constant mean function

m(x) = β
# Arguments:
* `β::Float64`: Constant
""" ->
type mCONST <: Mean
    β::Float64
    mCONST(β::Float64) = new(β)
end

@doc """
# Description
Constructor for the zero mean function

m(x) = 0
""" ->
mZERO() = mCONST(0.0)

meanf(mConst::mCONST,x::Matrix{Float64}) =  fill(mConst.β, size(x,2))

get_params(mConst::mCONST) = Float64[mConst.β]
num_params(mConst::mCONST) = 1
function set_params!(mConst::mCONST, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Constant mean function only has 1 parameter"))
    mConst.β = hyp[1]
end
function grad_meanf(mConst::mCONST, x::Vector{Float64})
    dM_theta = ones(size(x,2))
    return dM_theta
end
