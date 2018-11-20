# A class of Matern isotropic functions including the Matrern 3/2 and 5/2, where d= 3 or 5. Also the exponential function, where d=1
abstract type MaternIso <: Isotropic{Euclidean} end
abstract type MaternARD <: StationaryARD{WeightedEuclidean} end

@inline function dKij_dθp(mat::MaternARD, X::AbstractMatrix, i::Int, j::Int, p::Int, dim::Int)
    r=distij(metric(mat),X,i,j,dim)
    if p <= dim
        wdiffp=dist2ijk(metric(mat),X,i,j,p)
        if wdiffp > 0
            return dk_dll(mat,r,wdiffp)
        else
            return zero(dk_dll(mat,r,wdiffp))
        end
    elseif p==dim+1
        return dk_dlσ(mat, r)
    else
        return NaN
    end
end
@inline function dKij_dθp(mat::MaternARD, X::AbstractMatrix, data::StationaryARDData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(mat,X,i,j,p,dim)
end

@inline function dk_dθp(mat::MaternIso, r::Real, p::Int)
    if p==1
        return r == 0.0 ? 0.0 : dk_dll(mat, r)
    elseif p==2
        return dk_dlσ(mat, r)
    else
        return NaN
    end
end

include("mat12_iso.jl")
include("mat12_ard.jl")

include("mat32_iso.jl")
include("mat32_ard.jl")

include("mat52_iso.jl")
include("mat52_ard.jl")

"""
    Matern(ν::Real, ll::Union{Real,Vector{Real}}, lσ::Real)

Create Matérn kernel of type `ν` (i.e. `ν = 1/2`, `ν = 3/2`, or `ν = 5/2`) with length scale
`exp.(ll)` and signal standard deviation `exp(σ)`.

See also [`Mat12Iso`](@ref), [`Mat12Ard`](@ref), [`Mat32Iso`](@ref), [`Mat32Ard`](@ref),
[`Mat52Iso`](@ref), and [`Mat52Ard`](@ref).
"""
function Matern(ν::Real, ll::Real, lσ::Real)
    if ν==1/2
        kern = Mat12Iso(ll, lσ)
    elseif ν==3/2
        kern = Mat32Iso(ll, lσ)
    elseif ν==5/2
        kern = Mat52Iso(ll, lσ)
    else throw(ArgumentError("Only Matern 1/2, 3/2 and 5/2 are implementable"))
    end
    return kern
end

function Matern(ν::Real, ll::Vector{<:Real}, lσ::Real)
    if ν==1/2
        kern = Mat12Ard(ll, lσ)
    elseif ν==3/2
        kern = Mat32Ard(ll, lσ)
    elseif ν==5/2
        kern = Mat52Ard(ll, lσ)
    else throw(ArgumentError("Only Matern 1/2, 3/2 and 5/2 are implementable"))
    end
    return kern
end

@deprecate Mat Matern
