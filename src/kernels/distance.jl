# distance(k::Stationary, X::MatF64, Y::MatF64) = pairwise(metric(k), X, Y)
distance(k::Stationary, x::VecF64, y::VecF64) = evaluate(metric(k), x, y)

distance(k::Isotropic, X::MatF64, data::IsotropicData) = data.R
function distance(k::Stationary, X::MatF64)
    nobsv = size(X, 2)
    return distance!(Matrix{Float64}(undef, nobsv, nobsv), metric(k), X)
end
distance!(dist::MatF64, k::Stationary, X::MatF64) = distance!(dist, metric(k), X)
function distance!(dist::MatF64, m::PreMetric, X::MatF64)
    dim, nobsv = size(X)
    for i in 1:nobsv
        dist[i,i] = 0.0
        for j in 1:i-1
            dist[i,j] = dist[j,i] = distij(m, X, i, j, dim)
        end
    end
    return dist
end
function distance(k::Stationary, X::MatF64, Y::MatF64)
    nobsx = size(X, 2)
    nobsy = size(Y, 2)
    return distance!(Matrix{Float64}(undef, nobsx, nobsy), metric(k), X, Y)
end
distance!(dist::MatF64, k::Stationary, X::MatF64, Y::MatF64) = distance!(dist, metric(k), X, Y)
function distance!(dist::MatF64, m::PreMetric, X::MatF64, Y::MatF64)
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
function distance(k::StationaryARD, X::MatF64, data::StationaryARDData)
    distance(k, X)
end
@inline _SqEuclidean_ijk(X::MatF64,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2
@inline _SqEuclidean_ijk(X1::MatF64,X2::MatF64,i::Int,j::Int,k::Int)=(X1[k,i]-X2[k,j])^2
@inline function _SqEuclidean_ij(X::MatF64,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _SqEuclidean_ijk(X,i,j,k)
    end
    return s
end
@inline function _SqEuclidean_ij(X1::MatF64,X2::MatF64,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _SqEuclidean_ijk(X1,X2,i,j,k)
    end
    return s
end
# Squared Euclidean
# @inline distijk(dist::SqEuclidean,X::MatF64,i::Int,j::Int,k::Int)=_SqEuclidean_ijk(X,i,j,k)
# @inline distijk(dist::SqEuclidean,X1::MatF64,X2::MatF64,i::Int,j::Int,k::Int)=_SqEuclidean_ijk(X1,X2,i,j,k)
@inline distij(dist::SqEuclidean,X::MatF64,i::Int,j::Int,dim::Int)=_SqEuclidean_ij(X,i,j,dim)
@inline distij(dist::SqEuclidean,X1::MatF64,X2::MatF64,i::Int,j::Int,dim::Int)=_SqEuclidean_ij(X1,X2,i,j,dim)

# Euclidean
@inline distij(dist::Euclidean,X::MatF64,i::Int,j::Int,dim::Int)=√_SqEuclidean_ij(X,i,j,dim)
@inline distij(dist::Euclidean,X1::MatF64,X2::MatF64,i::Int,j::Int,dim::Int)=√_SqEuclidean_ij(X1,X2,i,j,dim)

@inline _WeightedSqEuclidean_ijk(weights::VecF64,X::MatF64,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2*weights[k]
@inline _WeightedSqEuclidean_ijk(weights::VecF64,X1::MatF64,X2::MatF64,i::Int,j::Int,k::Int)=(X1[k,i]-X2[k,j])^2*weights[k]
@inline function _WeightedSqEuclidean_ij(weights::VecF64,X::MatF64,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _WeightedSqEuclidean_ijk(weights,X,i,j,k)
    end
    return s
end
@inline function _WeightedSqEuclidean_ij(weights::VecF64,X1::MatF64,X2::MatF64,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _WeightedSqEuclidean_ijk(weights,X1,X2,i,j,k)
    end
    return s
end

# Weighted Squared Euclidean
@inline distijk(dist::WeightedSqEuclidean,X::MatF64,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X,i,j,k)
@inline distijk(dist::WeightedSqEuclidean,X1::MatF64,X2::MatF64,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X1,X2,i,j,k)
@inline distij(dist::WeightedSqEuclidean,X::MatF64,i::Int,j::Int,dim::Int)=_WeightedSqEuclidean_ij(dist.weights,X,i,j,dim)
@inline distij(dist::WeightedSqEuclidean,X1::MatF64,X2::MatF64,i::Int,j::Int,dim::Int)=_WeightedSqEuclidean_ij(dist.weights,X1,X2,i,j,dim)

# Weighted Euclidean
@inline dist2ijk(dist::WeightedEuclidean,X::MatF64,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X,i,j,k)
@inline dist2ijk(dist::WeightedEuclidean,X1::MatF64,X2::MatF64,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X1,X2,i,j,k)
@inline distij(dist::WeightedEuclidean,X::MatF64,i::Int,j::Int,dim::Int)=√_WeightedSqEuclidean_ij(dist.weights,X,i,j,dim)
@inline distij(dist::WeightedEuclidean,X1::MatF64,X2::MatF64,i::Int,j::Int,dim::Int)=√_WeightedSqEuclidean_ij(dist.weights,X1,X2,i,j,dim)
