# Periodic Function 

@doc """
# Description
Constructor for the Periodic kernel (covariance)

k(x,x') = σ²exp(-2sin²(π|x-x'|/p)/ℓ²)
# Arguments:
* `ll::Vector{Float64}`: Log of length scale ℓ
* `lσ::Float64`        : Log of the signal standard deviation σ
* `lp::Float64`        : Log of the period
""" ->
type Periodic <: Kernel
    ll::Float64      # Log of Length scale 
    lσ::Float64      # Log of signal std
    lp::Float64      # Log of period
    Periodic(ll::Float64, lσ::Float64, lp::Float64) = new(ll, lσ, lp)
end

function kern(peri::Periodic, x::Vector{Float64}, y::Vector{Float64})
    ℓ2 = exp(2.0*peri.ll)
    σ2 = exp(2.0*peri.lσ)
    p = exp(peri.lp)
    σ2*exp(-2.0/ℓ2*sin(π*euclidean(x,y)/p)^2)
end

get_params(peri::Periodic) = Float64[peri.ll, peri.lσ, peri.lp]
num_params(peri::Periodic) = 3

function set_params!(peri::Periodic, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Periodic function has only three parameters"))
    peri.ll, peri.lσ, peri.lp = hyp
end

function grad_kern(peri::Periodic, x::Vector{Float64}, y::Vector{Float64})
    ℓ = exp(peri.ll)
    σ2 = exp(2*peri.lσ)
    p      = exp(peri.lp)
    dxy = euclidean(x,y)
    
    dK_dℓ   = 4.0*σ2*(sin(pi*dxy/p)/ℓ)^2*exp(-2/ℓ^2*sin(pi*dxy/p)^2)
    dK_dσ = 2.0*σ2*exp(-2/ℓ^2*sin(pi*dxy/p)^2)
    dK_dp     = 4.0/ℓ^2*σ2*(pi*dxy/p)*sin(pi*dxy/p)*cos(pi*dxy/p)*exp(-2/ℓ^2*sin(pi*dxy/p)^2)
    dK_theta = [dK_dℓ, dK_dσ, dK_dp]
    return dK_theta
end

# This makes crossKern slower for some reason...

function crossKern(X::Matrix{Float64}, peri::Periodic)
    d, nobsv = size(X)
    ℓ2 = exp(2.0*peri.ll)
    σ2 = exp(2.0*peri.lσ)
    p = exp(peri.lp)
    R = pairwise(Euclidean(), X)
    for i in 1:1:nobsv, j in 1:i
        @inbounds R[i,j] =  σ2*exp(-2.0/ℓ2*sin(π*R[i,j]/p)^2)
        @inbounds R[j,i] = R[i,j]
    end
    return R
end

function grad_stack(X::Matrix{Float64}, peri::Periodic)
    d, nobsv = size(X)
    ℓ = exp(peri.ll)
    σ2 = exp(2*peri.lσ)
    p      = exp(peri.lp)
    dxy = pairwise(Euclidean(), X)
    
    stack = Array(Float64, nobsv, nobsv, 3)
    for i in 1:nobsv, j in 1:i
        @inbounds stack[i,j,1] = 4.0*σ2*(sin(pi*dxy[i,j]/p)/ℓ)^2*exp(-2/ℓ^2*sin(pi*dxy[i,j]/p)^2)  # dK_dℓ
        @inbounds stack[i,j,1] = stack[j,i,1]
        
        @inbounds stack[i,j,2] = 2.0*σ2*exp(-2/ℓ^2*sin(pi*dxy[i,j]/p)^2)        # dK_dσ
        @inbounds stack[i,j,2] = stack[j,i,2]
        
        @inbounds stack[i,j,3] = 4.0/ℓ^2*σ2*(pi*dxy[i,j]/p)*sin(pi*dxy[i,j]/p)*cos(pi*dxy[i,j]/p)*exp(-2/ℓ^2*sin(pi*dxy[i,j]/p)^2)    # dK_dp
        @inbounds stack[i,j,3] = stack[j,i,3]
    end

    return stack
end
