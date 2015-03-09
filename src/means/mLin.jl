#Linear mean function

type mLIN <: Mean
    β::Vector{Float64}
    mLIN(β::Vector{Float64}) = new(β)
end    
    
meanf(mLin::mLIN,x::Matrix{Float64}) =  x'*mLin.β

params(mLin::mLIN) = Vector{Float64}[mLin.β]
num_params(mLin::mLIN) = gp.dim
function set_params!(mLin::mLIN, hyp::Vector{Float64})
    length(hyp) == gp.dim || throw(ArgumentError("Linear mean function only has $(gp.dim) parameters"))
    mLin.β = hyp
end
function grad_meanf(mLin::mLIN, x::Vector{Float64})
    dM_theta = x
    return dM_theta
end    

