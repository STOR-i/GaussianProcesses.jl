#Linear ARD Covariance Function


@doc """
# Description
Constructor for the ARD linear kernel (covariance)

k(x,x') = xᵀL⁻²x', where L = diag(l₁,l₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale l
""" ->
type LINard <: Kernel
    ll::Vector{Float64}      # Log of Length scale
    dim::Int                 # Number of hyperparameters
    LINard(ll::Vector{Float64}) = new(ll,size(ll,1))
end

function kern(linArd::LINard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(linArd.ll)
    K = dot(x./ell,y./ell)
    return K
end

get_params(linArd::LINard) = [linArd.ll]
num_params(linArd::LINard) = linArd.dim

function set_params!(linArd::LINard, hyp::Vector{Float64})
    length(hyp) == linArd.dim || throw(ArgumentError("Linear ARD kernel has $(linArd.dim) parameters"))
    linArd.ll = hyp
end

function grad_kern(linArd::LINard, x::Vector{Float64}, y::Vector{Float64})
    ell = exp(linArd.ll)
    
    dK_ell = -2.0*(x./ell).*(y./ell)
    dK_theta = [dK_ell]
    return dK_theta
end
