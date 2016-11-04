# Subtypes of Stationary must define the following functions:
# metric(kernel::Stationary) = ::Metric
# cov(k::Stationary, r::Float64) = ::Float64
# grad_kern!

abstract Stationary <: Kernel
abstract StationaryData <: KernelData

distance{M<:MatF64}(k::Stationary, X::M) = pairwise(metric(k), X)
distance{M1<:MatF64,M2<:MatF64}(k::Stationary, X::M1, Y::M2) = pairwise(metric(k), X, Y)
distance{V1<:VecF64,V2<:VecF64}(k::Stationary, x::V1, y::V2) = evaluate(metric(k), x, y)

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

abstract Isotropic <: Stationary

type IsotropicData <: StationaryData
    R::Matrix{Float64}
end

function KernelData{M<:MatF64}(k::Isotropic, X::M)
     IsotropicData(distance(k, X))
end
function kernel_data_key{M<:MatF64}(k::Isotropic, X::M)
    return Symbol(:IsotropicData_, metric(k))
end

distance{M<:MatF64}(k::Isotropic, X::M, data::IsotropicData) = data.R
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

abstract StationaryARD <: Stationary

type StationaryARDData <: StationaryData
    dist_stack::Array{Float64, 3}
end

# May need to customized in subtypes
function KernelData{M<:MatF64}(k::StationaryARD, X::M)
    dim, nobsv = size(X)
    dist_stack = Array(Float64, nobsv, nobsv, dim)
    for d in 1:dim
        grad_ls = view(dist_stack, :, :, d)
        pairwise!(grad_ls, SqEuclidean(), view(X, d:d,:))
    end
    StationaryARDData(dist_stack)
end
kernel_data_key{M<:MatF64}(k::StationaryARD, X::M) = Symbol(:StationaryARDData, metric(k))

function distance{M<:MatF64}(k::StationaryARD, X::M, data::StationaryARDData)
    ### This commented section is slower than recalculating the distance from scratch...
    # nobsv = size(data.dist_stack,1)
    # d = length(k.ℓ2)
    # weighted = broadcast(/, data.dist_stack, reshape(k.ℓ2, (1,1,d)))
    # return reshape(sum(weighted, 3), (nobsv, nobsv))
    return pairwise(metric(k), X)
end
@inline distijk{M<:MatF64}(dist::SqEuclidean,X::M,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2
@inline function distij{M<:MatF64}(dist::SqEuclidean,X::M,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += distijk(dist,X,i,j,k)
    end
    return s
end
@inline distijk{M<:MatF64}(dist::Euclidean,X::M,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2
@inline function distij{M<:MatF64}(dist::Euclidean,X::M,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += distijk(dist,X,i,j,k)
    end
    return √s
end
@inline distijk{M<:MatF64}(dist::WeightedSqEuclidean,X::M,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2*dist.weights[k]
@inline function distij{M<:MatF64}(dist::WeightedSqEuclidean,X::M,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += distijk(dist,X,i,j,k)
    end
    return s
end
@inline dist2ijk{M<:MatF64}(dist::WeightedEuclidean,X::M,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2*dist.weights[k]
@inline function distij{M<:MatF64}(dist::WeightedEuclidean,X::M,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += dist2ijk(dist,X,i,j,k)
    end
    return √s
end
