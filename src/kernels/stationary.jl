# Subtypes of Stationary must define the following functions:
# metric(k::Stationary) = ::Metric
# cov(k::Stationary, r::Float64) = ::Float64
# grad_kern!

abstract Stationary <: Kernel

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

function cov(k::Stationary, x::Matrix{Float64})
    nobsv = size(x, 2)
    R = distance(k, x)
    for i in 1:nobsv, j in 1:i
        @inbounds R[i,j] = cov(k, R[i,j])
        @inbounds R[j,i] = R[i,j]
    end
    return R
end
