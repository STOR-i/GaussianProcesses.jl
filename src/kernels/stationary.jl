# Subtypes of Stationary must define the following functions:
# metric(kernel::Stationary) = ::Metric
# cov(k::Stationary, r::Float64) = ::Float64
# grad_kern!

abstract Stationary <: Kernel
abstract Isotropic <: Stationary
abstract StationaryARD <: Stationary

abstract StationaryData <: KernelData

type IsotropicData <: StationaryData
    R::Matrix{Float64}
end

type StationaryARDData <: StationaryData
    dist_stack::Array{Float64, 3}
end

distance(k::Stationary, X::Matrix{Float64}) = pairwise(metric(k), X)
distance(k::Stationary, X::Matrix{Float64}, Y::Matrix{Float64}) = pairwise(metric(k), X, Y)
distance(k::Stationary, x::Vector{Float64}, y::Vector{Float64}) = evaluate(metric(k), x, y)
distance(k::Isotropic, data::IsotropicData) = copy(data.R)

function distance(k::StationaryARD, data::StationaryARDData)
    nobsv = size(data.dist_stack,1)
    d = length(k.ℓ2)
    weighted = broadcast(/, data.dist_stack, reshape(k.ℓ2, (1,1,d)))
    return reshape(sum(weighted, 3), (nobsv, nobsv))
end


function KernelData(k::Isotropic, X::Matrix{Float64})
     IsotropicData(distance(k, X))
end

# May need to customized in subtypes
function KernelData(k::StationaryARD, X::Matrix{Float64})
    d, nobsv = size(X)
    dist_stack = Array(Float64, nobsv, nobsv, d)
    for i in 1:d
        grad_ls = view(dist_stack, :, :, i)
        pairwise!(grad_ls, SqEuclidean(), X[i,:])
    end
    StationaryARDData(dist_stack)
end

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
    R = distance(k, data)
    for i in 1:nobsv, j in 1:i
        @inbounds R[i,j] = cov(k, R[i,j])
        @inbounds R[j,i] = R[i,j]
    end
    return R
end
