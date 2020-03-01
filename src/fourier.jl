# using Debugger
mutable struct RFF
    dimension::Int64 # Number of Fourier features to compute
    α::AbstractArray
    β::AbstractArray
    ω::AbstractArray
    Ztr::AbstractArray
    L::AbstractArray
end

function RFF(m::Int64, X::AbstractArray; σ::Float64=1.0)
    n = size(X, 1)
    α = zeros(n, 1)
    ω = zeros(m, 1)
    β = zeros(m, n)
    Z = zeros(n, m)
    L = zeros(m, m)
    return RFF(m, α, β, ω, Z, L)
end

function _compute_ω(m::Int64, σ::Float64, X::AbstractArray, ω::AbstractArray, β::AbstractArray)
    N = size(X, 1)
    B = repeat(β, outer = [1, N]) # Should yield an M x N matrix
    norm = √(2/m)
    cos_inner =σ * (ω * X') # should be of m x size(X, 1)
    @assert size(cos_inner) == (m, N) "cos_inner of incorrect dimensions"
    Z = norm * cos.(σ * (ω * X') + B)
    return Z', ω, B, β
end

function compute_ω(m::Int64, σ::Float64, X::AbstractArray)
    N, D = size(X)
    ω = rand(Normal(), (m, D))
    β = rand(Uniform(0, 2π), (m, 1))
    return _compute_ω(m, σ, X, ω, β)
end
compute_ω(m::Int64, σ::GaussianProcesses.Scalar, X::AbstractArray) = compute_ω(m, convert(σ, Float64), X)

# """
#     scale(F, ℓ)
#
# Scale the frequencies by the current lengthscale value(s).
# """
# function reset_∇!(F::RFF)
#     F.∇mat = zeros(F.N, 2*F.M)
# end
#
# function scale!(F::RFF, ℓ::Float64)
#     F.ω = F.ω/ℓ
#     reset_∇!(F)
# end
#
# function scale!(F::RFF, ℓ::Array{Float64})
#     F.ω = F.ω./ℓ
#     reset_∇!(F)
# end
#
# scale(ω::AbstractArray, ℓ::Float64) = (1/ℓ)*ω
# scale(ω::AbstractArray, ℓ::AbstractArray) = (1/ℓ).*ω
#
# """
#     update(F)
#
# Update the Fourier approximating parameters
# """
# function update!(F::RFF, params::AbstractArray)
#     update_σ!(F, params[1])
#     update_ℓ!(F, params[2])
#     n_ω = size(f.ω, 1)
#     update_ω!(F, params[3:(2+n_ω)])
# end
#
# function update_ℓ!(F::RFF, p::Float64)
#     F.ℓ = p
#     scale!(F)
# end
#
# function update_ℓ!(F::RFF, p::Array{Float64})
#     F.ℓ = p
#     scale!(F)
# end
#
# function update_σ!(F::RFF, p::Float64)
#     F.σ = p
# end
#
# function update_σ!(F::RFF, p::AbstractArray)
#     F.σ = p
# end
# function update_ω!(F::RFF, p::AbstractArray)
#     F.ω = p
# end
# """
#     build_design_mat(F, X)
#
# Build the design matrix that is of dimension N x 2*M, where N is the number of obs. and M is the number of features.
# """
# # TODO: Might be best to have RFF and SSGP as one struct - can just load that in then as it'll contain, N, M, X and ω
# function build_design_mat(F::RFF, X::AbstractMatrix)
#     N = size(X, 2)
#     @info "nobs:" N
#     ϕ_x = zeros(N, 2*F.M) # Need 2*M rows for Cos and sin. Of shape N x 2m
#     @info "ϕ matrix" ϕ_x
#     ϕ_x[:, 1:F.M] = cos.(F.ω * X)' # RHS Should output an N x m matrix
#     ϕ_x[:, (F.M + 1):end] = sin.(F.ω * X)'
#     return ϕ_x
# end
