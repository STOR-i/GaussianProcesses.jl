#Constant mean function

type mCONST <: Mean
    β::Float64
    mCONST(β::Float64) = new(β)
end

mZERO() = mCONST(0.0)

meanf(mConst::mCONST,x::Matrix{Float64}) =  fill(mConst.β, size(x,2))

params(mConst::mCONST) = Float64[mConst.β]
num_params(mConst::mCONST) = gp.dim
function set_params!(mConst::mCONST, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Constant mean function only has 1 parameter"))
    mConst.β = hyp
end
function grad_meanf(mConst::mCONST, x::Vector{Float64})
    dM_theta = ones(size(x,2))
    return dM_theta
end
