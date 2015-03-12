# Linear mean function

type mLIN <: Mean
    β::Vector{Float64}
    dim::Int
    mLIN(β::Vector{Float64}) = new(β, length(β)
end
    
meanf(mLin::mLIN,x::Matrix{Float64}) =  x'*mLin.β

params(mLin::mLIN) = mLin.β
num_params(mLin::mLIN) = mLin.dim
function set_params!(mLin::mLIN, hyp::Vector{Float64})
    length(hyp) == mLin.dim || throw(ArgumentError("Linear mean function only has $(gp.dim) parameters"))
    mLin.β = hyp
end
function grad_meanf(mLin::mLIN, x::Vector{Float64})
    dM_theta = x
    return dM_theta
end
