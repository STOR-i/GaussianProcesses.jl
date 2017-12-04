# distance{M<:MatF64}(k::Stationary, X::M) = pairwise(metric(k), X)
distance{M1<:MatF64,M2<:MatF64}(k::Stationary, X::M1, Y::M2) = pairwise(metric(k), X, Y)
distance{V1<:VecF64,V2<:VecF64}(k::Stationary, x::V1, y::V2) = evaluate(metric(k), x, y)

# distance{V1<:VecF64,V2<:VecF64}(k::Stationary{Euclidean}, x::V1, y::V2) = √sum((xi^2+yi^2) for (xi,yi) in zip(xi,yi))

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
