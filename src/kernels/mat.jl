# A class of Matern isotropic functions including the Matrern 3/2 and 5/2, where d= 3 or 5. Also the exponential function, where d=1

type MAT <: Kernel
    d::Int           # Type of Matern function
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of Signal std
    MAT(d::Int,ll::Float64=0.0, lσ::Float64=0.0) = new(d,ll,lσ)
end

function kern(mat::MAT, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)

    if mat.d==1
        K = sigma2*exp(-norm(x-y)/ell)
    elseif mat.d==3
        K = sigma2*(1+sqrt(3)*norm(x-y)/ell)*exp(-sqrt(3)*norm(x-y)/ell)
    elseif mat.d==5
        K = sigma2*(1+sqrt(5)*norm(x-y)/ell+5*norm(x-y)^2/(3*ell^2))*exp(-sqrt(5)*norm(x-y)/ell)
    else throw(ArgumentError("For the Matern covariance d must equal 1, 3 or 5"))
    end
    return K
end

get_params(mat::MAT) = Float64[mat.ll, mat.lσ]
num_params(mat::MAT) = 2
function set_params!(mat::MAT, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern $(mat.d)/2 only has two parameters"))
    mat.ll, mat.lσ = hyp
end

function grad_kern(mat::MAT, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(mat.ll)
    sigma2 = exp(2*mat.lσ)

    if mat.d==1
         dK_ell = sigma2*norm(x-y)/ell*exp(-norm(x-y)/ell)
        dK_sigma = 2.0*sigma2*exp(-norm(x-y)/ell)
        dK_theta = [dK_ell,dK_sigma]
    elseif mat.d==3
        dK_ell = sigma2*(sqrt(3)*norm(x-y)/ell)^2*exp(-sqrt(3)*norm(x-y)/ell)
        dK_sigma = 2.0*sigma2*(1+sqrt(3)*norm(x-y)/ell)*exp(-sqrt(3)*norm(x-y)/ell)
        dK_theta = [dK_ell,dK_sigma]
    else mat.d==5
        dK_ell = sigma2*((5*norm(x-y)^2/ell^2)*(1+sqrt(5)*norm(x-y)/ell)/3)*exp(-sqrt(5)*norm(x-y)/ell)
        dK_sigma = 2.0*sigma2*(1+sqrt(5)*norm(x-y)/ell+5*norm(x-y)^2/(3*ell^2))*exp(-sqrt(5)*norm(x-y)/ell)
        dK_theta = [dK_ell,dK_sigma]
    end
    return dK_theta
end    
