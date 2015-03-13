#Linear Covariance Function

type LIN <: Kernel
    ll::Float64      # Log of Length scale 
    LIN(l::Float64=1.0) = new(ll)
end

function kern(lin::LIN, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(lin.ll)
    K = dot(x,y)/ell^2
    return K
end

params(lin::LIN) = Float64[lin.ll]
num_params(lin::LIN) = 1

function set_params!(lin::LIN, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Squared exponential only has one parameter"))
    lin.ll = hyp
end

function grad_kern(lin::LIN, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(lin.ll)
    
    dK_ell = -2.0*dot(x,y)/ell^2
    dK_theta = [dK_ell]
    return dK_theta
end
