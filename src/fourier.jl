# using Debugger
mutable struct RFF
    dimension::Int64
    ω::AbstractArray
    ωs::AbstractArray
    # Number of features
    M::Int64
    # Number of obs.
    N::Int64
    # Lengthscale
    ℓ::AbstractArray
    # Amplitude
    σ::Float64
    # Gradient matrix
    ∇mat::AbstractMatrix
end

function RFF(d::Int64, M::Int64, K::SEArd, N::Int64; amplitude::Float64=1.0)
    ω = rand(Normal(), M, d) # TODO: Should this be M or 2*M
    ℓ = ones(d)
    grad_mat = zeros(N, 2*M)
    return RFF(d, ω, scale(ω, ℓ), M, N, K.iℓ2, amplitude, grad_mat)
end

"""
    scale(F, ℓ)

Scale the frequencies by the current lengthscale value(s).
"""
function reset_∇!(F::RFF)
    F.∇mat = zeros(F.N, 2*F.M)
end

function scale!(F::RFF, ℓ::Float64)
    F.ω = F.ω/ℓ
    reset_∇!(F)
end

function scale!(F::RFF, ℓ::Array{Float64})
    F.ω = F.ω./ℓ
    reset_∇!(F)
end

scale(ω::AbstractArray, ℓ::Float64) = (1/ℓ)*ω
scale(ω::AbstractArray, ℓ::AbstractArray) = (1/ℓ).*ω

"""
    update(F)

Update the Fourier approximating parameters
"""
function update!(F::RFF, params::AbstractArray)
    update_σ!(F, params[1])
    update_ℓ!(F, params[2])
    n_ω = size(f.ω, 1)
    update_ω!(F, params[3:(2+n_ω)])
end

function update_ℓ!(F::RFF, p::Float64)
    F.ℓ = p
    scale!(F)
end

function update_ℓ!(F::RFF, p::Array{Float64})
    F.ℓ = p
    scale!(F)
end

function update_σ!(F::RFF, p::Float64)
    F.σ = p
end

function update_σ!(F::RFF, p::AbstractArray)
    F.σ = p
end
function update_ω!(F::RFF, p::AbstractArray)
    F.ω = p
end
"""
    build_design_mat(F, X)

Build the design matrix that is of dimension N x 2*M, where N is the number of obs. and M is the number of features.
"""
# TODO: Might be best to have RFF and SSGP as one struct - can just load that in then as it'll contain, N, M, X and ω
function build_design_mat(F::RFF, X::AbstractMatrix)
    N = size(X, 2)
    @info "nobs:" N
    ϕ_x = zeros(N, 2*F.M) # Need 2*M rows for Cos and sin. Of shape N x 2m
    @info "ϕ matrix" ϕ_x
    ϕ_x[:, 1:F.M] = cos.(F.ω * X)' # RHS Should output an N x m matrix
    ϕ_x[:, (F.M + 1):end] = sin.(F.ω * X)'
    return ϕ_x
end
