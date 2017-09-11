# Subtypes of Stationary must define the following functions:
# metric(kernel::Stationary) = ::Metric
# cov(k::Stationary, r::Float64) = ::Float64

@compat abstract type Stationary <:Kernel end
@compat abstract type StationaryData <: KernelData end

distance{M<:MatF64}(kern::Stationary, X::M) = pairwise(metric(kern), X)
distance{M1<:MatF64,M2<:MatF64}(kern::Stationary, X::M1, Y::M2) = pairwise(metric(kern), X, Y)
distance{V1<:VecF64,V2<:VecF64}(kern::Stationary, x::V1, y::V2) = evaluate(metric(kern), x, y)

cov{V1<:VecF64,V2<:VecF64}(kern::Stationary, x::V1, y::V2) = cov(kern, distance(kern, x, y))

function cov{M1<:MatF64,M2<:MatF64}(kern::Stationary, x1::M1, x2::M2)
    nobsv1 = size(x1, 2)
    nobsv2 = size(x2, 2)
    R = distance(kern, x1, x2)
    for i in 1:nobsv1
        for j in 1:nobsv2
            R[i,j] = cov(kern, R[i,j])
        end
    end
    return R
end

function cov{M<:MatF64}(kern::Stationary, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = copy(distance(kern, X, data))
    for i in 1:nobsv
        for j in 1:i
            @inbounds R[i,j] = cov(kern, R[i,j])
            @inbounds R[j,i] = R[i,j]
        end
    end
    return R
end
function cov!{M<:MatF64,K<:Stationary}(cK::MatF64, kern::K, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = distance(kern, X, data)
    for i in 1:nobsv
        @inbounds for j in 1:i
            cK[i,j] = cov(kern, R[i,j])
            cK[j,i] = cK[i,j]
        end
    end
    return cK
end
function addcov!{M<:MatF64}(s::MatF64, kern::Stationary, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = distance(kern, X, data)
    for j in 1:nobsv
        @simd for i in 1:j
            @inbounds s[i,j] += cov(kern, R[i,j])
            @inbounds s[j,i] = s[i,j]
        end
    end
    return R
end
function multcov!{M<:MatF64}(s::MatF64, kern::Stationary, X::M, data::StationaryData)
    nobsv = size(X, 2)
    R = distance(kern, X, data)
    for j in 1:nobsv
        @simd for i in 1:j
            @inbounds s[i,j] *= cov(kern, R[i,j])
            @inbounds s[j,i] = s[i,j]
        end
    end
    return R
end
@inline dk_dlθ(kern::Stationary, r::Float64, θ::Type{Val{:σ2}}) = 2.0*cov(kern,r)
@inline function dKij_dθp{M<:MatF64}(kern::Stationary, X::M, Xdim::Int64, 
                 i::Int64, j::Int64, 
                 θ::Type{Val{:σ2}}, θp::Int64, θdim::Int64, 
                 data::KernelData)
    return dk_dlθ(kern, distij(metric(kern),X,i,j,Xdim), θ)
end
@inline function dKij_dθp{M<:MatF64}(kern::Stationary, X::M, Xdim::Int64, 
                 i::Int64, j::Int64, 
                 θ::Type{Val{:σ2}}, θp::Int64, θdim::Int64)
    return dk_dlθ(kern, distij(metric(kern),X,i,j,Xdim), θ)
end

# Isotropic Kernels

@compat abstract type Isotropic <: Stationary end

type IsotropicData <: StationaryData
    R::Matrix{Float64}
end

function KernelData{M<:MatF64}(kern::Isotropic, X::M)
     IsotropicData(distance(kern, X))
end
function kernel_data_key{M<:MatF64}(kern::Isotropic, X::M)
    return @sprintf("%s_%s", "IsotropicData", metric(kern))
end

distance{M<:MatF64}(kern::Isotropic, X::M, data::IsotropicData) = data.R
function addcov!{M<:MatF64}(s::MatF64, kern::Isotropic, X::M, data::IsotropicData)
    nobsv = size(X, 2)
    R = distance(kern, X, data)
    for j in 1:nobsv
        @simd for i in 1:j
            @inbounds s[i,j] += cov(kern, R[i,j])
            @inbounds s[j,i] = s[i,j]
        end
    end
    return R
end
@inline function dKij_dθp{M<:MatF64}(kern::Isotropic,X::M,Xdim::Int,
                                     i::Int,j::Int,
                                     θ,θp::Int,θdim::Int)
    return dk_dlθ(kern, distij(metric(kern),X,i,j,Xdim), θ)
end
@inline function dKij_dθp{M<:MatF64}(kern::Isotropic,X::M,Xdim::Int,
                                     i::Int,j::Int,
                                     θ,θp::Int,θdim::Int,
                                     data::IsotropicData)
    return dk_dlθ(kern, data.R[i,j], θ)
end
@inline function dKij_dθp{M<:MatF64}(kern::Isotropic, X::M, Xdim::Int64, 
                 i::Int64, j::Int64, 
                 θ::Type{Val{:σ2}}, θp::Int64, θdim::Int64)
    return dk_dlθ(kern, distij(metric(kern),X,i,j,Xdim), θ)
end
@inline function dKij_dθp{M<:MatF64}(kern::Isotropic, X::M, Xdim::Int64, 
                 i::Int64, j::Int64, 
                 θ::Type{Val{:σ2}}, θp::Int64, θdim::Int64, 
                 data::IsotropicData)
    return dk_dlθ(kern, data.R[i,j], θ)
end

# StationaryARD Kernels

@compat abstract type StationaryARD <: Stationary end

type StationaryARDData <: StationaryData
    dist_stack::Array{Float64, 3}
end

# May need to customized in subtypes
function KernelData{M<:MatF64}(kern::StationaryARD, X::M)
    dim, nobsv = size(X)
    dist_stack = Array{Float64}( nobsv, nobsv, dim)
    for d in 1:dim
        grad_ls = view(dist_stack, :, :, d)
        pairwise!(grad_ls, SqEuclidean(), view(X, d:d,:))
    end
    StationaryARDData(dist_stack)
end
kernel_data_key{M<:MatF64}(kern::StationaryARD, X::M) = @sprintf("%s_%s", "StationaryARDData", metric(kern))

function distance{M<:MatF64}(kern::StationaryARD, X::M, data::StationaryARDData)
    ### This commented section is slower than recalculating the distance from scratch...
    # nobsv = size(data.dist_stack,1)
    # d = length(kern.ℓ2)
    # weighted = broadcast(/, data.dist_stack, reshape(kern.ℓ2, (1,1,d)))
    # return reshape(sum(weighted, 3), (nobsv, nobsv))
    return pairwise(metric(kern), X)
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
