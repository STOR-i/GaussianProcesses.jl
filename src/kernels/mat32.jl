# Matern 3/2 Function
type MAT32 <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of Signal std
    MAT32(l::Float64=1.0, σ::Float64=1.0) = new(log(l),log(σ))
end

function kern(mat32::MAT32, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat32.ll)
    sigma2 = exp(2*mat32.lσ)
    
    K = sigma2*(1+sqrt(3)*norm(x-y)/ell)*exp(-sqrt(3)*norm(x-y)/ell)
    return K
end    
params(mat32::MAT32) = exp(Float64[mat32.ll, mat32.lσ])
num_params(mat32::MAT32) = 2
function set_params!(mat32::MAT32, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 3/2 only has two parameters"))
    mat32.ll, mat32.lσ = hyp
end
function grad_kern(mat32::MAT32, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat32.ll)
    sigma2 = exp(2*mat32.lσ)
    
    dK_ell = sigma2*(sqrt(3)*norm(x-y)/ell)^2*exp(-sqrt(3)*norm(x-y)/ell)
    dK_sigma = 2.0*sigma2*(1+sqrt(3)*norm(x-y)/ell)*exp(-sqrt(3)*norm(x-y)/ell)
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end    
