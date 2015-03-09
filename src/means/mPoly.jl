#Polynomial mean function

type mPOLY <: Mean
    β::Vector{Float64}    #Polynomial coefficients
    d::Int64              #Polynomial degree
    mPOLY(β::Vector{Float64},d::Int64) = new(β,d)
end    
    
function meanf(mPoly::mPOLY,x::Matrix{Float64})
    dim, nobsv = size(x)
    z = zeros(nobsv,1)
    beta = reshape(mPoly.β,dim,mPoly.d)
    for i in 1:mPoly.d
        z = z + x'.^i*beta[:,i]
    end
    return z
end    

params(mPoly::mPOLY) = Float64[mPoly.β]
num_params(mPoly::mPOLY) = gp.dim
function set_params!(mPoly::mPOLY, hyp::Vector{Float64})
    length(hyp) == gp.dim || throw(ArgumentError("Polynomial mean function only has $(gp.dim) parameters"))
    mPoly.β = hyp
end

function grad_meanf(mPoly::mPOLY, x::Vector{Float64})
      nobsv = size(x,2)
      z = zeros(nobsv,1)
    for i in 1:mPoly.d
     z = z + x'.^i
    end
    dM_theta = z
    return dM_theta
end    

