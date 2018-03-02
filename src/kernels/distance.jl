distance{M1<:MatF64,M2<:MatF64}(k::Stationary, X::M1, Y::M2) = pairwise(metric(k), X, Y)
distance{V1<:VecF64,V2<:VecF64}(k::Stationary, x::V1, y::V2) = evaluate(metric(k), x, y)

distance{M<:MatF64}(k::Isotropic, X::M, data::IsotropicData) = data.R
function distance{M<:MatF64}(k::Stationary, X::M)
    dim, nobsv = size(X)
    m = metric(k)
    dist = Matrix{Float64}(nobsv,nobsv)
    for i in 1:nobsv
        dist[i,i] = 0.0
        for j in 1:i-1
            dist[i,j] = dist[j,i] = distij(m, X, i, j, dim)
        end
    end
    return dist
end
function distance{M<:MatF64}(k::StationaryARD, X::M, data::StationaryARDData)
    distance(k, X)
end
@inline _SqEuclidean_ijk{M<:MatF64}(X::M,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2
@inline _SqEuclidean_ijk{M1<:MatF64,M2<:MatF64}(X1::M1,X2::M2,i::Int,j::Int,k::Int)=(X1[k,i]-X2[k,j])^2
@inline function _SqEuclidean_ij{M<:MatF64}(X::M,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _SqEuclidean_ijk(X,i,j,k)
    end
    return s
end
@inline function _SqEuclidean_ij{M1<:MatF64,M2<:MatF64}(X1::M1,X2::M2,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _SqEuclidean_ijk(X1,X2,i,j,k)
    end
    return s
end
# Squared Euclidean
# @inline distijk{M<:MatF64}(dist::SqEuclidean,X::M,i::Int,j::Int,k::Int)=_SqEuclidean_ijk(X,i,j,k)
# @inline distijk{M1<:MatF64,M2<:MatF64}(dist::SqEuclidean,X1::M1,X2::M2,i::Int,j::Int,k::Int)=_SqEuclidean_ijk(X1,X2,i,j,k)
@inline distij{M<:MatF64}(dist::SqEuclidean,X::M,i::Int,j::Int,dim::Int)=_SqEuclidean_ij(X,i,j,dim)
@inline distij{M1<:MatF64,M2<:MatF64}(dist::SqEuclidean,X1::M1,X2::M2,i::Int,j::Int,dim::Int)=_SqEuclidean_ij(X1,X2,i,j,dim)

# Euclidean
@inline distij{M<:MatF64}(dist::Euclidean,X::M,i::Int,j::Int,dim::Int)=√_SqEuclidean_ij(X,i,j,dim)
@inline distij{M1<:MatF64,M2<:MatF64}(dist::Euclidean,X1::M1,X2::M2,i::Int,j::Int,dim::Int)=√_SqEuclidean_ij(X1,X2,i,j,dim)

@inline _WeightedSqEuclidean_ijk{V<:VecF64,M<:MatF64}(weights::V,X::M,i::Int,j::Int,k::Int)=(X[k,i]-X[k,j])^2*weights[k]
@inline _WeightedSqEuclidean_ijk{V<:VecF64,M1<:MatF64,M2<:MatF64}(weights::V,X1::M1,X2::M2,i::Int,j::Int,k::Int)=(X1[k,i]-X2[k,j])^2*weights[k]
@inline function _WeightedSqEuclidean_ij{V<:VecF64,M<:MatF64}(weights::V,X::M,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _WeightedSqEuclidean_ijk(weights,X,i,j,k)
    end
    return s
end
@inline function _WeightedSqEuclidean_ij{V<:VecF64,M1<:MatF64,M2<:MatF64}(weights::V,X1::M1,X2::M2,i::Int,j::Int,dim::Int)
    s = 0.0
    @inbounds @simd for k in 1:dim
        s += _WeightedSqEuclidean_ijk(weights,X1,X2,i,j,k)
    end
    return s
end

# Weighted Squared Euclidean
@inline distijk{M<:MatF64}(dist::WeightedSqEuclidean,X::M,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X,i,j,k)
@inline distijk{M1<:MatF64,M2<:MatF64}(dist::WeightedSqEuclidean,X1::M1,X2::M2,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X1,X2,i,j,k)
@inline distij{M<:MatF64}(dist::WeightedSqEuclidean,X::M,i::Int,j::Int,dim::Int)=_WeightedSqEuclidean_ij(dist.weights,X,i,j,dim)
@inline distij{M1<:MatF64,M2<:MatF64}(dist::WeightedSqEuclidean,X1::M1,X2::M2,i::Int,j::Int,dim::Int)=_WeightedSqEuclidean_ij(dist.weights,X1,X2,i,j,dim)

# Weighted Euclidean
@inline dist2ijk{M<:MatF64}(dist::WeightedEuclidean,X::M,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X,i,j,k)
@inline dist2ijk{M1<:MatF64,M2<:MatF64}(dist::WeightedEuclidean,X1::M1,X2::M2,i::Int,j::Int,k::Int)=_WeightedSqEuclidean_ijk(dist.weights,X1,X2,i,j,k)
@inline distij{M<:MatF64}(dist::WeightedEuclidean,X::M,i::Int,j::Int,dim::Int)=√_WeightedSqEuclidean_ij(dist.weights,X,i,j,dim)
@inline distij{M1<:MatF64,M2<:MatF64}(dist::WeightedEuclidean,X1::M1,X2::M2,i::Int,j::Int,dim::Int)=√_WeightedSqEuclidean_ij(dist.weights,X1,X2,i,j,dim)
