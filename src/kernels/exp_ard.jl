# Squared Exponential Function with ARD

@doc """
# Description
Constructor for the ARD Squared Exponential kernel (covariance)

k(x,x') = σ²exp(-(x-x')ᵀL⁻²(x-x')/2), where L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the inverexpk length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type ExpArd <: StationaryARD{WeightedSqEuclidean}
    iℓ2::Vector{Float64}      # Inverexpk squared Length scale
    σ2::Float64              # Signal variance
    priors::Array          # Array of priors for kernel parameters
    ExpArd(ll::Vector{Float64}, lσ::Float64) = new(exp.(-2.0*ll),exp(2.0*lσ),[])
end

function set_params!(expk::ExpArd, hyp::Vector{Float64})
    length(hyp) == num_params(expk) || throw(ArgumentError("ExpArd only has $(num_params(expk)) parameters"))
    d = length(expk.iℓ2)
    expk.iℓ2 = exp.(-2.0*hyp[1:d])
    expk.σ2 = exp(2.0*hyp[d+1])
end

get_params(expk::ExpArd) = [-log.(expk.iℓ2)/2.0; log(expk.σ2)/2.0]
get_param_names(k::ExpArd) = [get_param_names(k.iℓ2, :ll); :lσ]
num_params(expk::ExpArd) = length(expk.iℓ2) + 1

cov(expk::ExpArd, r::Float64) = expk.σ2*exp(-0.5*r)

@inline dk_dll(expk::ExpArd, r::Float64, wdiffp::Float64) = wdiffp*cov(expk,r)
@inline function dKij_dθp{M<:MatF64}(expk::ExpArd, X::M, i::Int, j::Int, p::Int, dim::Int)
    if p <= dim
        return dk_dll(expk, distij(metric(expk),X,i,j,dim), distijk(metric(expk),X,i,j,p))
    elseif p==dim+1
        return dk_dlσ(expk, distij(metric(expk),X,i,j,dim))
    else
        return NaN
    end
end
@inline function dKij_dθp{M<:MatF64}(expk::ExpArd, X::M, data::StationaryARDData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(expk,X,i,j,p,dim)
end
