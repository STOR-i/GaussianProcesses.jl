#Squared Exponential Function
type SE <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of Signal std
    SE(l::Float64=1.0, σ::Float64=1.0) = new(log(l),log(σ))
end

function kern(se::SE, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(se.ll)
    sigma2 = exp(2*se.lσ)
    
    K = sigma2*exp(-0.5*norm(x-y)^2/ell^2)
    return K
end    
params(se::SE) = exp(Float64[se.ll, se.lσ])
num_params(se::SE) = 2
function set_params!(se::SE, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Squared exponential only has two parameters"))
    se.ll, se.lσ = hyp
end
function grad_kern(se::SE, x::Vector{Float64}, y::Vector{Float64})
     ell = exp(se.ll)
     sigma2 = exp(2*se.lσ)
    
       dK_ell = sigma2*norm(x-y)^2/ell^2*exp(-0.5*norm(x-y)^2/ell^2)
       dK_sigma = 2.0*sigma2*exp(-0.5*norm(x-y)^2/ell^2)
    
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end    

