# Subtypes of Stationary must define the following functions:
# cov(k::Stationary, r::Float64) = ::Float64
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
function addcov!(cK::MatF64, k::Stationary, X1::MatF64, X2::MatF64)
    dim1, nobsv1 = size(X1)
    dim2, nobsv2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    nobsv1==size(cK,1) || throw(ArgumentError("X1 and cK incompatible nobsv"))
    nobsv2==size(cK,2) || throw(ArgumentError("X2 and cK incompatible nobsv"))
    dim = dim1
    met = metric(k)
    for i in 1:nobsv1
        for j in 1:nobsv2
            cK[i,j] += cov(k, distij(met, X1, X2, i, j, dim))
        end
    end
    return cK
end
function multcov!(cK::MatF64, k::Stationary, X1::MatF64, X2::MatF64)
    dim1, nobsv1 = size(X1)
    dim2, nobsv2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    nobsv1==size(cK,1) || throw(ArgumentError("X1 and cK incompatible nobsv"))
    nobsv2==size(cK,2) || throw(ArgumentError("X2 and cK incompatible nobsv"))
    dim = dim1
    met = metric(k)
    for i in 1:nobsv1
        for j in 1:nobsv2
            cK[i,j] *= cov(k, distij(met, X1, X2, i, j, dim))
        end
    end
    return cK
end
function Statistics.cov(k::Stationary, X1::MatF64, X2::MatF64)
    nobsv1 = size(X1, 2)
    nobsv2 = size(X2, 2)
    cK = Array{Float64}(undef, nobsv1, nobsv2)
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
function Statistics.cov(k::Stationary, X::MatF64, data::StationaryData)
    nobsv = size(X, 2)
    cK = Matrix{Float64}(undef, nobsv, nobsv)
    cov!(cK, k, X, data)
end
function Statistics.cov(k::Stationary, X::MatF64)
    nobsv = size(X, 2)
    cK = Matrix{Float64}(undef, nobsv, nobsv)
    cov!(cK, k, X)
end
function addcov!(cK::MatF64, k::Stationary, X::MatF64)
    dim, nobsv = size(X)
    nobsv==size(cK,1) || throw(ArgumentError("X and cK incompatible nobsv"))
    nobsv==size(cK,2) || throw(ArgumentError("X and cK incompatible nobsv"))
    met = metric(k)
    @inbounds for i in 1:nobsv
        for j in 1:i
            cK[i,j] += cov(k, distij(met, X, i, j, dim))
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function addcov!(cK::MatF64, k::Stationary, X::MatF64, d::StationaryData)
    addcov!(cK, k, X)
end
function multcov!(cK::MatF64, k::Stationary, X::MatF64)
    dim, nobsv = size(X)
    nobsv==size(cK,1) || throw(ArgumentError("X and cK incompatible nobsv"))
    nobsv==size(cK,2) || throw(ArgumentError("X and cK incompatible nobsv"))
    met = metric(k)
    @inbounds for i in 1:nobsv
        for j in 1:i
            cK[i,j] *= cov(k, distij(met, X, i, j, dim))
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function multcov!(cK::MatF64, k::Stationary, X::MatF64, data::StationaryData)
    multcov!(cK, k, X)
end
dk_dlσ(k::Stationary, r::Float64) = 2 * cov(k,r)

# Isotropic Kernels

abstract type Isotropic{D} <: Stationary{D} end

struct IsotropicData <: StationaryData
    R::Matrix{Float64}
end

function KernelData(k::Isotropic, X::MatF64)
     IsotropicData(distance(k, X))
end
function kernel_data_key(k::Isotropic, X::MatF64)
    return @sprintf("%s_%s", "IsotropicData", metric(k))
end

function addcov!(cK::MatF64, k::Isotropic, X::MatF64, data::IsotropicData)
    dim, nobsv = size(X)
    met = metric(k)
    for j in 1:nobsv
        @simd for i in 1:j
            @inbounds cK[i,j] += cov(k, distij(met, X, i, j, dim))
            @inbounds cK[j,i] = cK[i,j]
        end
    end
    return cK
end
@inline function dKij_dθp(kern::Isotropic,X::MatF64,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, distij(metric(kern),X,i,j,dim),p)
end
@inline function dKij_dθp(kern::Isotropic,X::MatF64,data::IsotropicData,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, data.R[i,j],p)
end
function grad_kern(kern::Isotropic, x::VecF64, y::VecF64)
    dist=distance(kern,x,y)
    return [dk_dθp(kern,dist,k) for k in 1:num_params(kern)]
end

# StationaryARD Kernels

abstract type StationaryARD{D} <: Stationary{D} end

struct StationaryARDData <: StationaryData
    dist_stack::Array{Float64, 3}
end

# May need to customized in subtypes
function KernelData(k::StationaryARD, X::MatF64)
    dim, nobsv = size(X)
    dist_stack = Array{Float64}(undef, nobsv, nobsv, dim)
    for d in 1:dim
        grad_ls = view(dist_stack, :, :, d)
        pairwise!(grad_ls, SqEuclidean(), view(X, d:d,:))
    end
    StationaryARDData(dist_stack)
end
kernel_data_key(k::StationaryARD, X::MatF64) = @sprintf("%s_%s", "StationaryARDData", metric(k))
