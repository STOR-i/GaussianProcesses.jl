# distance(k::Stationary, X::AbstractMatrix, Y::AbstractMatrix) = pairwise(metric(k), X, Y)
distance(k::Stationary, x::AbstractVector, y::AbstractVector) = evaluate(metric(k), x, y)

distance(k::Stationary, X::AbstractMatrix, data::EmptyData) = distance(k, X)
distance(k::Isotropic, X::AbstractMatrix, data::IsotropicData) = data.R
function distance(k::Stationary, X::AbstractMatrix)
    nobsv = size(X, 2)
    return distance!(Matrix{eltype(X)}(undef, nobsv, nobsv), metric(k), X)
end
distance!(dist::AbstractMatrix, k::Stationary, X::AbstractMatrix) = distance!(dist, metric(k), X)
function distance!(dist::AbstractMatrix, m::PreMetric, X::AbstractMatrix)
    dim, nobsv = size(X)
    for i in 1:nobsv
        dist[i,i] = 0.0
        for j in 1:i-1
            dist[i,j] = dist[j,i] = distij(m, X, i, j, dim)
        end
    end
    return dist
end
function distance(k::Stationary, X::AbstractMatrix, Y::AbstractMatrix)
    nobsx = size(X, 2)
    nobsy = size(Y, 2)
    return distance!(Matrix{eltype(Y)}(undef, nobsx, nobsy), metric(k), X, Y)
end
distance!(dist::AbstractMatrix, k::Stationary, X::AbstractMatrix, Y::AbstractMatrix) = distance!(dist, metric(k), X, Y)
function distance!(dist::AbstractMatrix, m::PreMetric, X::AbstractMatrix, Y::AbstractMatrix)
    dimx, nobsx = size(X)
    dimy, nobsy = size(Y)
    dimx == dimy || error("size(X, 1) != size(Y, 1)")
    for i in 1:nobsx
        for j in 1:nobsy
            dist[i,j] = distij(m, X, Y, i, j, dimx)
        end
    end
    return dist
end
function distance(k::StationaryARD, X::AbstractMatrix, data::StationaryARDData)
    distance(k, X)
end
@inline _SqEuclidean_ijk(X::AbstractMatrix,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2
@inline _SqEuclidean_ijk(X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,k::Int)=(X1[k,i]-X2[k,j])^2
@inline function _SqEuclidean_ij(X::AbstractMatrix,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _SqEuclidean_ijk(X,i,j,k)
    end
    return s
end
@inline function _SqEuclidean_ij(X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _SqEuclidean_ijk(X1,X2,i,j,k)
    end
    return s
end
# Squared Euclidean
# @inline distijk(dist::SqEuclidean,X::AbstractMatrix,i::Int,j::Int,k::Int)=_SqEuclidean_ijk(X,i,j,k)
# @inline distijk(dist::SqEuclidean,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,k::Int)=_SqEuclidean_ijk(X1,X2,i,j,k)
@inline distij(dist::SqEuclidean,X::AbstractMatrix,i::Int,j::Int,dim::Int)=_SqEuclidean_ij(X,i,j,dim)
@inline distij(dist::SqEuclidean,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,dim::Int)=_SqEuclidean_ij(X1,X2,i,j,dim)

# Euclidean
@inline distij(dist::Euclidean,X::AbstractMatrix,i::Int,j::Int,dim::Int)=√_SqEuclidean_ij(X,i,j,dim)
@inline function distij(dist::Euclidean,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,dim::Int)
    if X1 === X2 && i == j
        zero(eltype(X2))
    else
        √_SqEuclidean_ij(X1,X2,i,j,dim)
    end
end

@inline _WeightedSqEuclidean_ijk(weights::AbstractVector,X::AbstractMatrix,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2*weights[k]
@inline _WeightedSqEuclidean_ijk(weights::AbstractVector,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,k::Int)=(X1[k,i]-X2[k,j])^2*weights[k]
@inline function _WeightedSqEuclidean_ij(weights::AbstractVector,X::AbstractMatrix,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _WeightedSqEuclidean_ijk(weights,X,i,j,k)
    end
    return s
end
@inline function _WeightedSqEuclidean_ij(weights::AbstractVector,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _WeightedSqEuclidean_ijk(weights,X1,X2,i,j,k)
    end
    return s
end

# Weighted Squared Euclidean
@inline distijk(dist::WeightedSqEuclidean,X::AbstractMatrix,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X,i,j,k)
@inline distijk(dist::WeightedSqEuclidean,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X1,X2,i,j,k)
@inline distij(dist::WeightedSqEuclidean,X::AbstractMatrix,i::Int,j::Int,dim::Int)=_WeightedSqEuclidean_ij(dist.weights,X,i,j,dim)
@inline distij(dist::WeightedSqEuclidean,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,dim::Int)=_WeightedSqEuclidean_ij(dist.weights,X1,X2,i,j,dim)

# Weighted Euclidean
@inline dist2ijk(dist::WeightedEuclidean,X::AbstractMatrix,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X,i,j,k)
@inline dist2ijk(dist::WeightedEuclidean,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X1,X2,i,j,k)
@inline distij(dist::WeightedEuclidean,X::AbstractMatrix,i::Int,j::Int,dim::Int)=√_WeightedSqEuclidean_ij(dist.weights,X,i,j,dim)
@inline function distij(dist::WeightedEuclidean,X1::AbstractMatrix,X2::AbstractMatrix,i::Int,j::Int,dim::Int)
    if X1 === X2 && i == j
        zero(eltype(X2))
    else
        √_WeightedSqEuclidean_ij(dist.weights,X1,X2,i,j,dim)
    end
end
