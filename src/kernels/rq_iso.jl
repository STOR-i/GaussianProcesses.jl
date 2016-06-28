# Rational Quadratic Isotropic Covariance Function 

@doc """
# Description
Constructor for the isotropic Rational Quadratic kernel (covariance)

k(x,x') = σ²(1+(x-x')ᵀ(x-x')/2αℓ²)^{-α}
# Arguments:
* `ll::Float64`: Log of length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
* `lα::Float64`: Log of shape parameter α
""" ->
type RQIso <: Isotropic
    ℓ2::Float64      # Length scale 
    σ2::Float64      # Signal std
    α::Float64       # shape parameter
    RQIso(ll::Float64, lσ::Float64, lα::Float64) = new(exp(2*ll), exp(2*lσ), exp(lα))
end

function set_params!(rq::RQIso, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Rational Quadratic function has three parameters"))
    rq.ℓ2, rq.σ2, rq.α = exp(2.0*hyp[1]), exp(2.0*hyp[2]), exp(hyp[3])
end

get_params(rq::RQIso) = Float64[log(rq.ℓ2)/2.0, log(rq.σ2)/2.0, log(rq.α)]
get_param_names(rq::RQIso) = [:ll, :lσ, :lα]
num_params(rq::RQIso) = 3

metric(rq::RQIso) = SqEuclidean()
cov(rq::RQIso, r::Float64) = rq.σ2*(1.0+r/(2.0*rq.α*rq.ℓ2)).^(-rq.α)


function grad_kern(rq::RQIso, x::Vector{Float64}, y::Vector{Float64})
    r = distance(rq, x, y)
    
    g1 = rq.σ2*(r/rq.ℓ2)*(1.0+r/(2*rq.α*rq.ℓ2))^(-rq.α-1.0)       # dK_d(log ℓ)
    g2 = 2.0*rq.σ2*(1+r/(2*rq.α*rq.ℓ2))^(-rq.α)                   # dK_d(log σ)
    part = (1.0+r/(2*rq.α*rq.ℓ2))
    g3 = rq.σ2*part^(-rq.α)*(r/(2*rq.ℓ2*part)-rq.α*log(part))*rq.α  # dK_d(log α)
    return [g1,g2,g3]
end


function grad_stack!(stack::AbstractArray, rq::RQIso, X::Matrix{Float64}, data::IsotropicData)
    nobsv = size(X,2)
    R = distance(rq, X, data)
    
    for i in 1:nobsv, j in 1:i
        # Check these derivatives!
        @inbounds stack[i,j,1] = rq.σ2*((R[i,j])/rq.ℓ2)*(1.0+(R[i,j])/(2*rq.α*rq.ℓ2))^(-rq.α-1.0)  # dK_d(log ℓ)dK_dℓ
        @inbounds stack[i,j,2] = 2.0*rq.σ2*(1+(R[i,j])/(2*rq.α*rq.ℓ2))^(-rq.α) # dK_d(log σ)

        part = (1.0+R[i,j]/(2*rq.α*rq.ℓ2))
        @inbounds stack[i,j,3] = rq.σ2*part^(-rq.α)*((R[i,j])/(2*rq.ℓ2*part)-rq.α*log(part))*rq.α  # dK_d(log α)
        
        @inbounds stack[j,i,1] = stack[i,j,1]
        @inbounds stack[j,i,3] = stack[i,j,3]
        @inbounds stack[j,i,2] = stack[i,j,2]
    end
    return stack
end
