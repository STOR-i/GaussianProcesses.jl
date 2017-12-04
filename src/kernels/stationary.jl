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

function cov{M1<:MatF64,M2<:MatF64}(k::Stationary, x1::M1, x2::M2)
    nobsv1 = size(x1, 2)
    nobsv2 = size(x2, 2)
    R = distance(k, x1, x2)
    for i in 1:nobsv1
        for j in 1:nobsv2
            R[i,j] = cov(k, R[i,j])
        end
    end
    return R
end

function cov{M<:MatF64}(k::Stationary, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = copy(distance(k, X, data))
    for i in 1:nobsv
        for j in 1:i
            @inbounds R[i,j] = cov(k, R[i,j])
            @inbounds R[j,i] = R[i,j]
        end
    end
    return R
end
function cov!{M<:MatF64}(cK::MatF64, k::Stationary, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = distance(k, X, data)
    for i in 1:nobsv
        for j in 1:i
            @inbounds cK[i,j] = cov(k, R[i,j])
            @inbounds cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function addcov!{M<:MatF64}(s::MatF64, k::Stationary, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = distance(k, X, data)
    for j in 1:nobsv
        @simd for i in 1:j
            @inbounds s[i,j] += cov(k, R[i,j])
            @inbounds s[j,i] = s[i,j]
        end
    end
    return R
end
function multcov!{M<:MatF64}(s::MatF64, k::Stationary, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = distance(k, X, data)
    for j in 1:nobsv
        @simd for i in 1:j
            @inbounds s[i,j] *= cov(k, R[i,j])
            @inbounds s[j,i] = s[i,j]
        end
    end
    return R
end
dk_dlσ(k::Stationary, r::Float64) = 2.0*cov(k,r)

# Isotropic Kernels

@compat abstract type Isotropic{D} <: Stationary{D} end

type IsotropicData <: StationaryData
    R::Matrix{Float64}
end

function KernelData{M<:MatF64}(k::Isotropic, X::M)
     IsotropicData(distance(k, X))
end
function kernel_data_key{M<:MatF64}(k::Isotropic, X::M)
    return @sprintf("%s_%s", "IsotropicData", metric(k))
end

function addcov!{M<:MatF64}(s::MatF64, k::Isotropic, X::M, data::IsotropicData)
    nobsv = size(X, 2)
    R = distance(k, X, data)
    for j in 1:nobsv
        @simd for i in 1:j
            @inbounds s[i,j] += cov(k, R[i,j])
            @inbounds s[j,i] = s[i,j]
        end
    end
    return R
end
@inline function dKij_dθp{M<:MatF64}(kern::Isotropic,X::M,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, distij(metric(kern),X,i,j,dim),p)
end
@inline function dKij_dθp{M<:MatF64}(kern::Isotropic,X::M,data::IsotropicData,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, data.R[i,j],p)
end
function grad_kern{V1<:VecF64,V2<:VecF64}(kern::Isotropic, x::V1, y::V2)
    dist=distance(kern,x,y)
    return [dk_dθp(kern,dist,k) for k in 1:num_params(kern)]
end

# StationaryARD Kernels

@compat abstract type StationaryARD{D} <: Stationary{D} end

type StationaryARDData <: StationaryData
    dist_stack::Array{Float64, 3}
end

# May need to customized in subtypes
function KernelData{M<:MatF64}(k::StationaryARD, X::M)
    dim, nobsv = size(X)
    dist_stack = Array{Float64}( nobsv, nobsv, dim)
    for d in 1:dim
        grad_ls = view(dist_stack, :, :, d)
        pairwise!(grad_ls, SqEuclidean(), view(X, d:d,:))
    end
    StationaryARDData(dist_stack)
end
kernel_data_key{M<:MatF64}(k::StationaryARD, X::M) = @sprintf("%s_%s", "StationaryARDData", metric(k))

