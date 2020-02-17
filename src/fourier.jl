mutable struct RFF
    dimension::Int64
    ω::AbstractMatrix
    ωs::AbstractMatrix
    M::Int64
    ℓ::AbstractArray
    σ::Float64
    ∇mat::AbstractMatrix
end

function RFF(d::Int64, M::Int64, K::SEArd, N::Int64; amplitude::Float64=1.0)
    ω = rand(Normal(), M, d) # TODO: Should this be M or 2*M
    ℓ = ones(d)
    grad_mat = zeros(N, 2*M)
    return RFF(d, ω, scale_ω(ω, ℓ), M, K.iℓ2, amplitude, grad_mat)
end

mutable struct SSGPR
    X::AbstractMatrix
    y::AbstractArray
    fourier::RFF
    noise::Float64
    N::Int64
    d::Int64
    ∇mat::AbstractMatrix
end

function SSGPR(X::AbstractMatrix, y::AbstractArray, K::Kernel; M::Int64=20, noise::Float64=1.0)
    d = size(X, 2)
    N = size(X, 1)
    features = RFF(d, M, K, N)
    ∇mat = zeros(N, 2*M)
    GP = SSGPR(X, y, features, noise, N, d, ∇mat)
end

function scale_ω(ω::AbstractMatrix, ℓ::AbstractArray)
    ωscaled = (1/ℓ).*ω
    return ωscaled
end

# TODO: Possibly just pass a scalar value or a SSGPR struct
function build_design_mat(F::RFF, X::AbstractMatrix)
    N = size(X, 1)
    ϕ_x = zeros(N, 2*F.M) # Need 2*M rows for Cos and sin
    ϕ_x[:, 1:F.M] = sin.(X*transpose(F.ω))
    ϕ_x[:, (F.M + 1):end] = cos.(X*transpose(F.ω))
    return ϕ_x
end

function ∇l(X, gp::SSGPR)
    N = size(X, 0)

end
