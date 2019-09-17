using Optim, GaussianProcesses, Distributions, Random, RDatasets
Random.seed!(203617)

function scale_ω(ω::AbstractMatrix, ℓ::AbstractArray)
    ωscaled = (1/ℓ).*ω
    return ωscaled
end

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
    grad_mat = Array{Float64}(undef, N, 2*M)
    return RFF(d, ω, scale_ω(ω, ℓ), M, K.iℓ2, amplitude, grad_mat)
end

mutable struct SSGPR
    X::AbstractMatrix
    y::AbstractArray
    fourier::RFF
    noise::Float64
    N::Int64
    d::Int64
end

function SSGPR(X::AbstractMatrix, y::AbstractArray, K::Kernel; M::Int64=20, noise::Float64=1.0)
    d = size(X, 2)
    N = size(X, 1)
    features = RFF(d, M, K, N)
    GP = SSGPR(X, y, features, noise, N, d)
end

# TODO: Possibly just pass a scalar value or a SSGPR struct
function build_design_mat(F::RFF, X::AbstractMatrix)
    N = size(X, 1)
    ϕ_x = zeros(N, 2*F.M) # Need 2*M rows for Cos and sin
    ϕ_x[:, 1:F.M] = sin.(X*transpose(F.ω))
    ϕ_x[:, (F.M + 1):end] = cos.(X*transpose(F.ω))
    return ϕ_x
end

function grad_l(X, gp::SSGPR)
    N = size(X, 0)
    if gp.fourier.∇mat


# function

crabs = dataset("MASS","crabs");              # load the data
crabs = crabs[shuffle(1:size(crabs)[1]), :];  # shuffle the data

train = crabs[1:div(end,2),:];

X = convert(Matrix,train[:,7:end]);          # predictors
ybool = Array{Bool}(undef,size(train)[1])*1.0       # response
y = 1.5*cos.(ybool) + (1/3)*sin.(2*ybool) + rand(Normal(0, 0.1), length(ybool))

mZero = MeanZero();                # Zero mean function
kern = SE(zeros(2), 0.0);   # Matern 3/2 ARD kernel (note that hyperparameters are on the log scale)

gp = SSGPR(X, y, kern)
ϕ = build_design_mat(gp.fourier, X)

# gpe = GP(X', y, mZero, kern)       #Fit the GP
# optimize!(gpe; method=ConjugateGradient())   # Optimise the hyperparameters
# contour(gpe)
