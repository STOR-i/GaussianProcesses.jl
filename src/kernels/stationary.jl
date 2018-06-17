# Subtypes of Stationary must define the following functions:
# cov(k::Stationary, r::Float64) = ::Float64
# grad_kern!

@compat abstract type Stationary{D} <: Kernel where D <: Distances.SemiMetric end
@compat abstract type StationaryData <: KernelData end

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

cov{V1<:VecF64,V2<:VecF64}(k::Stationary, x::V1, y::V2) = cov(k, distance(k, x, y))
@inline function cov_ij(k::K, X::MatF64, i::Int, j::Int, dim::Int) where {K<:Stationary}
    cov(k, distij(metric(k), X, i, j, dim))
end
@inline function cov_ij(k::Stationary, X::MatF64, data::KernelData, i::Int, j::Int, dim::Int)
    cov(k, distij(metric(k), X, i, j, dim))
end
function cov!{M1<:MatF64,M2<:MatF64}(cK::MatF64, k::Stationary, X1::M1, X2::M2)
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
function cov(k::Stationary, X1::MatF64, X2::MatF64)
    nobsv1 = size(X1, 2)
    nobsv2 = size(X2, 2)
    cK = Array{Float64}(nobsv1, nobsv2)
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
            cK[i,j] = cov(k, distij(met, X, i, j, dim))
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function cov!(cK::MatF64, k::Stationary, X::MatF64, data::StationaryData)
    cov!(cK, k, X)
end
function cov(k::Stationary, X::MatF64, data::StationaryData)
    nobsv = size(X, 2)
    cK = Matrix{Float64}(nobsv, nobsv)
    cov!(cK, k, X, data)
end
function cov(k::Stationary, X::MatF64)
    nobsv = size(X, 2)
    cK = Matrix{Float64}(nobsv, nobsv)
    cov!(cK, k, X)
end
dk_dlσ(k::Stationary, r::Float64) = 2.0*cov(k,r)

# Isotropic Kernels

@compat abstract type Isotropic{D} <: Stationary{D} end

type IsotropicData <: StationaryData
    R::Matrix{Float64}
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

@compat abstract type StationaryARD{D} <: Stationary{D} end

type StationaryARDData <: StationaryData
    dist_stack::Array{Float64, 3}
end

# May need to customized in subtypes
function KernelData(k::StationaryARD, X::MatF64)
    dim, nobsv = size(X)
    dist_stack = Array{Float64}( nobsv, nobsv, dim)
    for d in 1:dim
        grad_ls = view(dist_stack, :, :, d)
        pairwise!(grad_ls, SqEuclidean(), view(X, d:d,:))
    end
    StationaryARDData(dist_stack)
end
kernel_data_key(k::StationaryARD, X::MatF64) = @sprintf("%s_%s", "StationaryARDData", metric(k))

