# Subtypes of Stationary must define the following functions:
# cov(k::Stationary, r::Number) = ::Float64
# grad_kern!

abstract type Stationary{D} <: Kernel where D <: Distances.SemiMetric end
abstract type StationaryData <: KernelData end

const _sqeuclidean = SqEuclidean()
const _euclidean = Euclidean()
@inline metric(kernel::Stationary{SqEuclidean}) = _sqeuclidean
@inline metric(kernel::Stationary{Euclidean}) = _euclidean
@inline function metric(kernel::Stationary{D}) where D <: Distances.SemiMetric
    return D()
end
@inline function metric(kernel::Stationary{WeightedSqEuclidean})
    return WeightedSqEuclidean(ard_weights(kernel))
end
@inline function metric(kernel::Stationary{WeightedEuclidean})
    return WeightedEuclidean(ard_weights(kernel))
end
@inline ard_weights(kernel::Stationary{WeightedSqEuclidean}) = kernel.iℓ2
@inline ard_weights(kernel::Stationary{WeightedEuclidean}) = kernel.iℓ2

cov(k::Stationary, x::AbstractVector, y::AbstractVector) = cov(k, distance(k, x, y))
@inline function cov_ij(k::Stationary, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int)
    cov(k, distij(metric(k), X1, X2, i, j, dim))
end
@inline function cov_ij(k::Stationary, X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int)
    cov(k, distij(metric(k), X1, X2, i, j, dim))
end
@inline function cov_ij(k::Stationary, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int)
    cov(k, distij(metric(k), X1, X2, i, j, dim))
end
function cov!(cK::AbstractMatrix, k::Stationary, X1::AbstractMatrix, X2::AbstractMatrix)
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
function cov(k::Stationary, X1::AbstractMatrix, X2::AbstractMatrix)
    nobsv1 = size(X1, 2)
    nobsv2 = size(X2, 2)
    cK = Array{eltype(X2)}(undef, nobsv1, nobsv2)
    cov!(cK, k, X1, X2)
    return cK
end

function cov!(cK::AbstractMatrix, k::Stationary, X::AbstractMatrix)
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
function cov!(cK::AbstractMatrix, k::Stationary, X::AbstractMatrix, data::StationaryData)
    cov!(cK, k, X)
end
function cov(k::Stationary, X::AbstractMatrix, data::StationaryData)
    nobsv = size(X, 2)
    cK = Matrix{eltype(X)}(undef, nobsv, nobsv)
    cov!(cK, k, X, data)
end
function cov(k::Stationary, X::AbstractMatrix)
    nobsv = size(X, 2)
    cK = Matrix{eltype(X)}(undef, nobsv, nobsv)
    cov!(cK, k, X)
end
dk_dlσ(k::Stationary, r::Float64) = 2 * cov(k,r)

# Isotropic Kernels

abstract type Isotropic{D} <: Stationary{D} end

struct IsotropicData{D} <: StationaryData
    R::D
end

function KernelData(k::Isotropic, X1::AbstractMatrix, X2::AbstractMatrix)
	 IsotropicData(distance(k, X1, X2))
end
function kernel_data_key(k::Isotropic, X1::AbstractMatrix, X2::AbstractMatrix)
	return @sprintf("%s_%s", "IsotropicData", metric(k))
end

@inline @inbounds function cov_ij(kern::Isotropic, X1::AbstractMatrix, X2::AbstractMatrix, data::IsotropicData, i::Int, j::Int, dim::Int)
    return cov(kern, data.R[i,j])
end
@inline function dKij_dθp(kern::Isotropic,X::AbstractMatrix,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, distij(metric(kern),X,i,j,dim), p)
end
@inline @inbounds function dKij_dθp(kern::Isotropic,X::AbstractMatrix,data::IsotropicData,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, data.R[i,j], p)
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Isotropic, X::AbstractMatrix, i::Int, j::Int, dim::Int, npars::Int)
    r = distij(metric(kern),X,i,j,dim)
    for p in 1:npars
        dK[p] = dk_dθp(kern, r, p)
    end
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Isotropic, X::AbstractMatrix, data::IsotropicData, 
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
function KernelData(k::StationaryARD, X1::AbstractMatrix, X2::AbstractMatrix)
	dim1, n1 = size(X1)
	dim2, n2 = size(X2)
	@assert dim1 == dim2
	dim = dim1
    dist_stack = Array{eltype(X2)}(undef, n1, n2, dim)
	for d in 1:dim1
		grad_ls = view(dist_stack, :, :, d)
		distance!(grad_ls, SqEuclidean(), view(X1, d:d,:), view(X2, d:d,:))
	end
	StationaryARDData(dist_stack)
end
kernel_data_key(k::StationaryARD, X1::AbstractMatrix, X2::AbstractMatrix) = @sprintf("%s_%s", "StationaryARDData", metric(k))
