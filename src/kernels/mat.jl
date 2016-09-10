# A class of Matern isotropic functions including the Matrern 3/2 and 5/2, where d= 3 or 5. Also the exponential function, where d=1
abstract MaternIso <: Isotropic
abstract MaternARD <: StationaryARD

@inline function dKij_dθp(mat::MaternARD, X::Matrix{Float64}, i::Int, j::Int, p::Int, dim::Int)
    r=distij(metric(mat),X,i,j,dim)
    if p <= dim
        wdiffp=dist2ijk(metric(mat),X,i,j,p)
        return wdiffp==0.0?0.0:dk_dll(mat,r,wdiffp)
    elseif p==dim+1
        return dk_dlσ(mat, r)
    else
        return NaN
    end
end
@inline function dKij_dθp(mat::MaternARD, X::Matrix{Float64}, data::StationaryARDData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(mat,X,i,j,p,dim)
end

@inline function dk_dθp(mat::MaternIso, r::Float64, p::Int)
    if p==1
        return r==0.0?0.0:dk_dll(mat, r)
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

@doc """
# Description
Constructor the Matern kernel, where ν defines the Matern type (i.e. ν = 1/2, 3/2 or 5/2).

# See also
Mat12Iso, Mat12Ard, Mat32Iso, Mat32Ard, Mat52Iso, Mat52Ard
""" ->
function Mat(ν::Float64,ll::Float64, lσ::Float64)
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

function Mat(ν::Float64,ll::Vector{Float64}, lσ::Float64)
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

