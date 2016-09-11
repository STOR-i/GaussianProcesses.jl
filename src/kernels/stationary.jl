# Subtypes of Stationary must define the following functions:
# metric(kernel::Stationary) = ::Metric
# cov(k::Stationary, r::Float64) = ::Float64
# grad_kern!

abstract Stationary <: Kernel
abstract StationaryData <: KernelData

distance(k::Stationary, X::Matrix{Float64}) = pairwise(metric(k), X)
distance(k::Stationary, X::Matrix{Float64}, Y::Matrix{Float64}) = pairwise(metric(k), X, Y)
distance(k::Stationary, x::Vector{Float64}, y::Vector{Float64}) = evaluate(metric(k), x, y)

cov(k::Stationary, x::Vector{Float64}, y::Vector{Float64}) = cov(k, distance(k, x, y))

function cov(k::Stationary, x1::Matrix{Float64}, x2::Matrix{Float64})
    nobsv1 = size(x1, 2)
    nobsv2 = size(x2, 2)
    R = distance(k, x1, x2)
    for i in 1:nobsv1, j in 1:nobsv2
        R[i,j] = cov(k, R[i,j])
    end
    return R
end

function cov(k::Stationary, X::Matrix{Float64}, data::StationaryData)
    nobsv = size(X, 2)
    R = copy(distance(k, X, data))
    for i in 1:nobsv, j in 1:i
        @inbounds R[i,j] = cov(k, R[i,j])
        @inbounds R[j,i] = R[i,j]
    end
    return R
end
function cov!(cK::AbstractMatrix{Float64}, k::Stationary, X::Matrix{Float64}, data::StationaryData)
    nobsv = size(X, 2)
    R = distance(k, X, data)
    for i in 1:nobsv, j in 1:i
        @inbounds cK[i,j] = cov(k, R[i,j])
        @inbounds cK[j,i] = cK[i,j]
    end
    return cK
end
function addcov!(s::AbstractMatrix{Float64}, k::Stationary, X::Matrix{Float64}, data::StationaryData)
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
function multcov!(s::AbstractMatrix{Float64}, k::Stationary, X::Matrix{Float64}, data::StationaryData)
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

# Isotropic Kernels

abstract Isotropic <: Stationary

type IsotropicData <: StationaryData
    R::Matrix{Float64}
end

function KernelData(k::Isotropic, X::Matrix{Float64})
     IsotropicData(distance(k, X))
end
function kernel_data_key(k::Isotropic, X::Matrix{Float64})
    return Symbol(:IsotropicData_, metric(k))
end

distance(k::Isotropic, X::Matrix{Float64}, data::IsotropicData) = data.R
function addcov!(s::AbstractMatrix{Float64}, k::Isotropic, X::Matrix{Float64}, data::IsotropicData)
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
@inline function dKij_dθp(kern::Isotropic,X::Matrix{Float64},i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, distij(metric(kern),X,i,j,dim),p)
end
@inline function dKij_dθp(kern::Isotropic,X::Matrix{Float64},data::IsotropicData,i::Int,j::Int,p::Int,dim::Int)
    return dk_dθp(kern, data.R[i,j],p)
end
function grad_kern(kern::Isotropic, x::Vector{Float64}, y::Vector{Float64})
    dist=distance(kern,x,y)
    return [dk_dθp(kern,dist,k) for k in 1:num_params(kern)]
end

# StationaryARD Kernels

abstract StationaryARD <: Stationary

type StationaryARDData <: StationaryData
    dist_stack::Array{Float64, 3}
end

# May need to customized in subtypes
function KernelData(k::StationaryARD, X::Matrix{Float64})
    dim, nobsv = size(X)
    dist_stack = Array(Float64, nobsv, nobsv, dim)
    for d in 1:dim
        grad_ls = view(dist_stack, :, :, d)
        pairwise!(grad_ls, SqEuclidean(), view(X, d:d,:))
    end
    StationaryARDData(dist_stack)
end
kernel_data_key(k::StationaryARD, X::Matrix{Float64}) = Symbol(:StationaryARDData, metric(k))

function distance(k::StationaryARD, X::Matrix{Float64}, data::StationaryARDData)
    ### This commented section is slower than recalculating the distance from scratch...
    # nobsv = size(data.dist_stack,1)
    # d = length(k.ℓ2)
    # weighted = broadcast(/, data.dist_stack, reshape(k.ℓ2, (1,1,d)))
    # return reshape(sum(weighted, 3), (nobsv, nobsv))
    return pairwise(metric(k), X)
end
@inline distijk(dist::SqEuclidean,X::Matrix{Float64},i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2
@inline function distij(dist::SqEuclidean,X::Matrix{Float64},i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += distijk(dist,X,i,j,k)
    end
    return s
end
@inline distijk(dist::Euclidean,X::Matrix{Float64},i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2
@inline function distij(dist::Euclidean,X::Matrix{Float64},i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += distijk(dist,X,i,j,k)
    end
    return √s
end
@inline distijk(dist::WeightedSqEuclidean,X::Matrix{Float64},i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2*dist.weights[k]
@inline function distij(dist::WeightedSqEuclidean,X::Matrix{Float64},i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += distijk(dist,X,i,j,k)
    end
    return s
end
@inline dist2ijk(dist::WeightedEuclidean,X::Matrix{Float64},i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2*dist.weights[k]
@inline function distij(dist::WeightedEuclidean,X::Matrix{Float64},i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += dist2ijk(dist,X,i,j,k)
    end
    return √s
end
dk_dlσ(k::Stationary, r::Float64) = 2.0*cov(k,r)
