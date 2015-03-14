# Matern 5/2 Function
type MAT52 <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of signal std
    MAT52(ll::Float64=0.0, lσ::Float64=0.0) = new(ll, lσ)
end

function kern(mat52::MAT52, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat52.ll)
    sigma2 = exp(2*mat52.lσ)

    K = sigma2*(1+sqrt(5)*norm(x-y)/ell+5*norm(x-y)^2/(3*ell^2))*exp(-sqrt(5)*norm(x-y)/ell)
    return K
end

params(mat52::MAT52) = Float64[mat52.ll, mat52.lσ]
num_params(mat52::MAT52) = 2

function set_params!(mat52::MAT52, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 5/2 only has two parameters"))
    mat52.ll, mat52.lσ = hyp
end

function grad_kern(mat52::MAT52, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat52.ll)
    sigma2 = exp(2*mat52.lσ)
    
    dK_ell = sigma2*((5*norm(x-y)^2/ell^2)*(1+sqrt(5)*norm(x-y)/ell)/3)*exp(-sqrt(5)*norm(x-y)/ell)
    dK_sigma = 2.0*sigma2*(1+sqrt(5)*norm(x-y)/ell+5*norm(x-y)^2/(3*ell^2))*exp(-sqrt(5)*norm(x-y)/ell)
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end
