using Optim, GaussianProcesses, Distributions, Random


Random.seed!(203617)
n=10;                          #number of training points
d = 2
x = 2π * rand(n, 2);              #predictors
y = sin.(x[:, 1]) + cos.(2*x[:, 2]) + 0.05*randn(n);   #regressors
kern = SE([0.0, 0.0], 0.0) 

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
end


mutable struct SSGPR
    X::AbstractMatrix
    y::AbstractArray    
    fourier::RFF
end

function RFF(d::Int64, M::Int64, K::SEArd; amplitude::Float64=1.0)
    ω = rand(Normal(), M, d)
    ℓ = ones(d)
    return RFF(d, ω, scale_ω(ω, ℓ), M, K.iℓ2, amplitude)
end

function SSGPR(X::AbstractMatrix, y::AbstractArray, K::Kernel; M::Int64=100)
    d = size(X, 2)
    features = RFF(d, M, K)
    GP = SSGPR(X, y, features)
end


# TODO: Possibly just pass a scalar value or a SSGPR struct
function build_design_mat!(F::RFF, X::AbstractMatrix)
    N = size(X, 1)
    ϕ_x = zeros(N, 2*F.M) # Need 2*M rows for Cos and sin
    ϕ_x[:, 1:F.M] = sin.(X*transpose(F.ω))
    ϕ_x[:, (F.M + 1):end] = cos.(X*transpose(F.ω))
    return ϕ_x    
end



gp = SSGPR(x, y, kern)
println(gp)

