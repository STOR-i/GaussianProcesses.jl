# A class of Matern ARD functions including the Matrern 3/2 and 5/2, where d= 3 or 5. Also the exponential function, where d=1

type MATard <: Kernel
    d::Int                 # Type of Matern function
    ll::Vector{Float64}    # Log of Length scale 
    lσ::Float64            # Log of Signal std
    dim::Int               # Number of hyperparameters
    MATard(d::Int,ll::Vector{Float64}, lσ::Float64) = new(d,ll,lσ,size(ll,1)+1)
end

function kern(matArd::MATard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(matArd.ll)
    sigma2 = exp(2*matArd.lσ)

    if matArd.d==1
        K = sigma2*exp(-norm((x-y)./ell))
    elseif matArd.d==3
        K = sigma2*(1+sqrt(3)*norm((x-y)./ell))*exp(-sqrt(3)*norm((x-y)./ell))
    elseif matArd.d==5
        K = sigma2*(1+sqrt(5)*norm((x-y)./ell)+5/3*norm((x-y)./ell)^2)*exp(-sqrt(5)*norm((x-y)./ell))
    else throw(ArgumentError("For the Matern covariance d must equal 1, 3 or 5"))
    end
    return K
end

params(matArd::MATard) = [matArd.ll, matArd.lσ]
num_params(matArd::MATard) = matArd.dim

function set_params!(matArd::MATard, hyp::Vector{Float64})
    length(hyp) == matArd.dim || throw(ArgumentError("Matern $(matArd.d)/2 only has $(matArd.dim) parameters"))
    matArd.ll = hyp[1:matArd.dim-1]
    matArd.lσ = hyp[matArd.dim]
end

function grad_kern(matArd::MATard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(matArd.ll)
    sigma2 = exp(2*matArd.lσ)

    if matArd.d==1
        dK_ell = sigma2.*((x-y)./ell).*exp(-norm((x-y)./ell))
        dK_sigma = 2.0*sigma2*exp(-norm((x-y)./ell))
        dK_theta = [dK_ell,dK_sigma]
    elseif matArd.d==3
        dK_ell = sigma2.*(sqrt(3).*((x-y)./ell).^2).*exp(-sqrt(3).*norm((x-y)./ell))
        dK_sigma = 2.0*sigma2*(1+sqrt(3)*norm((x-y)./ell))*exp(-sqrt(3)*norm((x-y)./ell))
        dK_theta = [dK_ell,dK_sigma]
    else matArd.d==5
        dK_ell = sigma2.*((5.*((x-y)./ell).^2).*(1+sqrt(5).*((x-y)./ell))/3).*exp(-sqrt(5).*norm((x-y)./ell))
        dK_sigma = 2.0*sigma2*(1+sqrt(5)*norm((x-y)./ell)+5/3*norm((x-y)./ell)^2)*exp(-sqrt(5)*norm((x-y)./ell))
        dK_theta = [dK_ell,dK_sigma]
    end
    return dK_theta
end    
