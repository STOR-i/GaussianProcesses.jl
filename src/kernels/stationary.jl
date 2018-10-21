# Subtypes of Stationary must define the following functions:
# cov(k::Stationary, r::Number) = ::Float64
# grad_kern!

abstract type Stationary{D} <: Kernel where D <: Distances.SemiMetric end
abstract type StationaryData <: KernelData end

function metric(kernel::Stationary{D}) where D <: Distances.SemiMetric
    return D()
end
function metric(kernel::Stationary{WeightedSqEuclidean})
    return WeightedSqEuclidean(ard_weights(kernel))
end
function metric(kernel::Stationary{WeightedEuclidean})
    return WeightedEuclidean(ard_weights(kernel))
end
ard_weights(kernel::Stationary{WeightedSqEuclidean}) = kernel.iℓ2
ard_weights(kernel::Stationary{WeightedEuclidean}) = kernel.iℓ2

Statistics.cov(k::Stationary, x::VecF64, y::VecF64) = cov(k, distance(k, x, y))
@inline function cov_ij(k::Stationary, X::MatF64, i::Int, j::Int, dim::Int)
    cov(k, distij(metric(k), X, i, j, dim))
end
@inline function cov_ij(k::Stationary, X::MatF64, data::KernelData, i::Int, j::Int, dim::Int)
    cov(k, distij(metric(k), X, i, j, dim))
end
function cov!(cK::MatF64, k::Stationary, X1::MatF64, X2::MatF64)
    dim1, nobsv1 = size(X1)
    dim2, nobsv2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    nobsv1==size(cK,1) || throw(ArgumentError("X1 and cK incompatible nobsv"))
    nobsv2==size(cK,2) || throw(ArgumentError("X2 and cK incompatible nobsv"))
    dim = dim1
    met = metric(k)
    for i in 1:nobsv1
        for j in 1:nobsv2
            cK[i,j] = cov(k, distij(met, X1, X2, i, j, dim))
        end
    end
    return cK
end
function Statistics.cov(k::Stationary, X1::MatF64, X2::AbstractArray{T, 2}) where T
    nobsv1 = size(X1, 2)
    nobsv2 = size(X2, 2)
    cK = Array{T}(undef, nobsv1, nobsv2)
    cov!(cK, k, X1, X2)
    return cK
end

function cov!(cK::MatF64, k::Stationary, X::MatF64)
    dim, nobsv = size(X)
    nobsv==size(cK,1) || throw(ArgumentError("X and cK incompatible nobsv"))
    nobsv==size(cK,2) || throw(ArgumentError("X and cK incompatible nobsv"))
    met = metric(k)
    @inbounds for i in 1:nobsv
        for j in 1:i
            cK[i, j] = cK[j, i] = cov(k, distij(met, X, i, j, dim))
        end
    end
    return cK
end
function cov!(cK::MatF64, k::Stationary, X::MatF64, data::StationaryData)
    cov!(cK, k, X)
end
function Statistics.cov(k::Stationary, X::MatF64, data::StationaryData)
    nobsv = size(X, 2)
    cK = Matrix{Float64}(undef, nobsv, nobsv)
    cov!(cK, k, X, data)
end
function Statistics.cov(k::Stationary, X::AbstractArray{T, 2}) where T
    nobsv = size(X, 2)
    cK = Matrix{T}(undef, nobsv, nobsv)
    cov!(cK, k, X)
end
dk_dlσ(k::Stationary, r::Float64) = 2 * cov(k,r)

# Isotropic Kernels

abstract type Isotropic{D} <: Stationary{D} end

struct IsotropicData{D} <: StationaryData
    R::D
end

function KernelData(k::Isotropic, X::MatF64)
     IsotropicData(distance(k, X))
end
function kernel_data_key(k::Isotropic, X::MatF64)
    return @sprintf("%s_%s", "IsotropicData", metric(k))
end

@inline @inbounds function cov_ij(kern::Isotropic, X::MatF64, data::IsotropicData, i::Int, j::Int, dim::Int)
    return cov(kern, data.R[i,j])
end
@inline function dKij_dθp(kern::Isotropic,X::MatF64,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, distij(metric(kern),X,i,j,dim), p)
end
@inline @inbounds function dKij_dθp(kern::Isotropic,X::MatF64,data::IsotropicData,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, data.R[i,j], p)
end
@inline @inbounds function dKij_dθ!(dK::VecF64, kern::Isotropic, X::MatF64, i::Int, j::Int, dim::Int, npars::Int)
    r = distij(metric(kern),X,i,j,dim)
    for p in 1:npars
        dK[p] = dk_dθp(kern, r, p)
    end
end
@inline @inbounds function dKij_dθ!(dK::VecF64, kern::Isotropic, X::MatF64, data::IsotropicData, 
                                    i::Int, j::Int, dim::Int, npars::Int)
    r = data.R[i,j]
    for iparam in 1:npars
        dK[iparam] = dk_dθp(kern, r, iparam)
    end
end

# StationaryARD Kernels

abstract type StationaryARD{D} <: Stationary{D} end

struct StationaryARDData{D} <: StationaryData
    dist_stack::D
end

# May need to customized in subtypes
function KernelData(k::StationaryARD, X::MatF64)
    dim, nobsv = size(X)
    dist_stack = Array{Float64}(undef, nobsv, nobsv, dim)
    for d in 1:dim
        grad_ls = view(dist_stack, :, :, d)
        distance!(grad_ls, SqEuclidean(), view(X, d:d,:))
    end
    StationaryARDData(dist_stack)
end
kernel_data_key(k::StationaryARD, X::MatF64) = @sprintf("%s_%s", "StationaryARDData", metric(k))
