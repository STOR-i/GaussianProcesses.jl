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
@inline function cov_ij(k::Stationary, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int)
    cov(k, distij(metric(k), X1, X2, i, j, dim))
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
@inline function dKij_dθp(kern::Isotropic,X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, distij(metric(kern),X1, X2,i,j,dim), p)
end
@inline @inbounds function dKij_dθp(kern::Isotropic,X1::AbstractMatrix, X2::AbstractMatrix,data::IsotropicData,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, data.R[i,j], p)
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Isotropic, X1::AbstractMatrix, X2::AbstractMatrix, data::EmptyData, i::Int, j::Int, dim::Int, npars::Int)
    r = distij(metric(kern),X1, X2,i,j,dim)
    for p in 1:npars
        dK[p] = dk_dθp(kern, r, p)
    end
end
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Isotropic, X1::AbstractMatrix, X2::AbstractMatrix, 
                                    data::IsotropicData, i::Int, j::Int, dim::Int, npars::Int)
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
