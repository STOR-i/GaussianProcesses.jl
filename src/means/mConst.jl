#Constant mean function

type mCONST <: Mean
    β::Float64
    mCONST(β::Float64=0.0) = new(β)
end    
    
meanf(mConst::mCONST,x::Matrix{Float64}) =  mConst.β*ones(size(x, 2))

params(mConst::mCONST) = Float64[mConst.β]
num_params(mConst::mCONST) = gp.dim
function set_params!(mConst::mCONST, hyp::Vector{Float64})
    length(hyp) == gp.dim || throw(ArgumentError("Constant mean function only has $(gp.dim) parameters"))
    mConst.β = hyp
end
function grad_meanf(mConst::mCONST, x::Vector{Float64})
    dM_theta = ones(size(x,2))
    return dM_theta
end    

