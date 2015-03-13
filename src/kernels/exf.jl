# Exponential Function
type EXF <: Kernel
    ll::Float64     #Log of length scale
    lσ::Float64     #Log of signal std
    EXF(ll::Float64=1.0, lσ::Float64=1.0) = new(ll,lσ)
end

function kern(exf::EXF, x::Vector{Float64},y::Vector{Float64})
    ell = exp(exf.ll)
    sigma2 = exp(2*exf.lσ)

    K = sigma2*exp(-norm(x-y)/ell)
    return K
end

params(exf::EXF) = exp(Float64[exf.ll, exf.lσ])
num_params(exf::EXF) = 2

function set_params!(exf::EXF, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Exponential covariance function only has two parameters"))
    exf.ll, exf.lσ = hyp
end

function grad_kern(exf::EXF, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(exf.ll)
    sigma2 = exp(2*exf.lσ)
    
    dK_ell = sigma2*norm(x-y)/ell*exp(-norm(x-y)/ell)
    dK_sigma = 2.0*sigma2*exp(-norm(x-y)/ell)
    dK_theta = [dK_ell,dK_sigma]
    return dK_theta
end


