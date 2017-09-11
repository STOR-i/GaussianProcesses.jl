# Squared Exponential Function with ARD

@doc """
# Description
Constructor for the ARD Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀL⁻²(x-x')/2), where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the inverse length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type SEArd <: StationaryARD
    params::@NT(iℓ2,σ2){PositiveGPVector,PositiveGPParam}
    priors::Array          # Array of priors for kernel parameters
end
function SEArd(iℓ2::Vector{Float64}, σ2::Float64)
    params = @NT(iℓ2,σ2){PositiveGPVector,PositiveGPParam}(
            PositiveGPVector(iℓ2),
            PositiveGPParam(σ2)
            )
    return SEArd(params, [])
end
@inline metric(se::SEArd) = WeightedSqEuclidean(get_value(se.params.iℓ2))
@inline cov(se::SEArd, r::Float64) = get_value(se.params.σ2)*exp(-0.5*r)

@inline dk_dll(se::SEArd, r::Float64, wdiffp::Float64) = wdiffp*cov(se,r)
@inline function dKij_dθp{M<:MatF64}(se::SEArd, X::M, Xdim::Int, i::Int, j::Int, 
                          θ::Type{Val{:iℓ2}}, θp::Int, θdim::Int)
    dk_dll(se, 
           distij(metric(se),X,i,j,Xdim), 
           distijk(metric(se),X,i,j,θp))
end
@inline function dKij_dθp{M<:MatF64}(se::SEArd, 
                                     X::M, Xdim::Int, i::Int, j::Int, 
                                     θ::Type{Val{:iℓ2}}, θp::Int, θdim::Int,
                                     data::StationaryARDData)
    # ignore data
    return dKij_dθp(se,X,Xdim,i,j,θ,θp,θdim)
end
