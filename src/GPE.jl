# Main GaussianProcess type

mutable struct GPE{X<:MatF64,Y<:VecF64,M<:Mean,K<:Kernel,P<:AbstractPDMat,D<:KernelData} <: GPBase
    # Observation data
    "Input observations"
    x::X
    "Output observations"
    y::Y

    # Model
    "Mean object"
    mean::M
    "Kernel object"
    kernel::K
    "Log standard deviation of observation noise"
    logNoise::Float64

    # Auxiliary data
    "Dimension of inputs"
    dim::Int
    "Number of observations"
    nobs::Int
    "Auxiliary observation data (to speed up calculations)"
    data::D
    "`(k + exp(2*obsNoise))`"
    cK::P
    "`(k + exp(2*obsNoise))⁻¹y`"
    alpha::Vector{Float64}
    "Marginal log-likelihood"
    mll::Float64
    "Log target (marginal log-likelihood + log priors)"
    target::Float64
    "Gradient of marginal log-likelihood"
    dmll::Vector{Float64}
    "Gradient of log-target (gradient of marginal log-likelihood + gradient of log priors)"
    dtarget::Vector{Float64}

    function GPE{X,Y,M,K,P}(x::X, y::Y, mean::M, kernel::K, logNoise::Float64) where {X,Y,M,K,P}
        dim, nobs = size(x)
        length(y) == nobs || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        data = KernelData(kernel, x)
        gp = new{X,Y,M,K,P,typeof(data)}(x, y, mean, kernel, logNoise, dim, nobs, data)
        initialise_target!(gp)
    end
end

"""
    GPE(x, y, mean, kernel[, logNoise])

Fit a Gaussian process to a set of training points. The Gaussian process is defined in
terms of its user-defined mean and covariance (kernel) functions. As a default it is
assumed that the observations are noise free.

# Arguments:
- `x::AbstractVecOrMat{Float64}`: Input observations
- `y::AbstractVector{Float64}`: Output observations
- `mean::Mean`: Mean function
- `kernel::Kernel`: Covariance function
- `logNoise::Float64`: Natural logarithm of the standard deviation for the observation
  noise. The default is -2.0, which is equivalent to assuming no observation noise.
"""
function GPE(x::MatF64, y::VecF64, mean::Mean, kernel::Kernel, logNoise::Float64 = -2.0, elastic::Bool = false) 
    if elastic
        x = ElasticArray(x)
        y = ElasticArray(y)
    end
    GPE{typeof(x),typeof(y),typeof(mean),typeof(kernel),(elastic ? ElasticPDMat : PDMat){Float64,Matrix{Float64}}}(x, y, mean, kernel, logNoise)
end

GPE(x::VecF64, y::VecF64, mean::Mean, kernel::Kernel, logNoise::Float64 = -2.0, elastic::Bool = false) =
    GPE(x', y, mean, kernel, logNoise, elastic)

"""
    GPE(; mean::Mean = MeanZero(), kernel::Kernel = SE(0.0, 0.0), logNoise::Float64 = -2.0)

Construct a [GPE](@ref) object without observations.
"""
function GPE(; mean::Mean = MeanZero(), kernel::Kernel = SE(0.0, 0.0), logNoise::Float64 = -2.0, elastic = false) 
    x = Array{Float64}(undef, 1, 0) # ElasticArrays don't like length(x) = 0.
    y = Array{Float64}(undef, 0)
    GPE(x, y, mean, kernel, logNoise, elastic)
end

"""
    GP(x, y, mean::Mean, kernel::Kernel[, logNoise::Float64=-2.0])

Fit a Gaussian process that is defined by its `mean`, its `kernel`, and the logarithm
`logNoise` of the standard deviation of its observation noise to a set of training points
`x` and `y`.

See also: [`GPE`](@ref)
"""
GP(x::AbstractVecOrMat{Float64}, y::VecF64, mean::Mean, kernel::Kernel,
   logNoise::Float64 = -2.0) = GPE(x, y, mean, kernel, logNoise)

"""
    fit!(gp::GPE{X,Y}, x::X, y::Y)

Fit Gaussian process `GPE` to a training data set consisting of input observations `x` and
output observations `y`.
"""
function fit!(gp::GPE{X,Y}, x::X, y::Y) where {X,Y}
    length(y) == size(x,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.x = x
    gp.y = y
    gp.data = KernelData(gp.kernel, x)
    gp.dim, gp.nobs = size(x)
    initialise_target!(gp)
end

fit!(gp::GPE, x::VecF64, y::VecF64) = fit!(gp, x', y)

import Base.append!
append!(gp, x::AbstractArray{Float64, 1}, y::Float64) = append!(gp, reshape(x, :, 1), [y])
function append!(gp::GPE{X,Y,M,K,P,D}, x::AbstractArray{Float64, 2}, y::AbstractArray{Float64, 1}) where {X,Y,M,K,P <: ElasticPDMat, D}
    newcov = [cov(gp.kernel, gp.x, x); cov(gp.kernel, x, x) + (exp(2*gp.logNoise) + 1e-5)*I]
    append!(gp.x, x)
    append!(gp.cK, newcov)
    gp.nobs += length(y)
    append!(gp.y, y)
    yy = gp.y - mean(gp.mean, gp.x) # would not need to be recomputed every time
    gp.alpha = gp.cK \ yy
    gp.mll = -(dot(yy, gp.alpha) + logdet(gp.cK) + log2π * gp.nobs) / 2
    gp.target = gp.mll   + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel)
    gp
end

#———————————————————————————————————————————————————————————
#Fast memory allocation function

"""
    get_ααinvcKI!(ααinvcKI::Matrix{Float64}, cK::AbstractPDMat, α::Vector)

Write `ααᵀ - cK * eye(nobs)` to `ααinvcKI` avoiding any memory allocation, where `cK` and
`α` are the covariance matrix and the alpha vector of a Gaussian process, respectively.
Hereby `α` is defined as `cK \\ (Y - μ)`.
"""
function get_ααinvcKI!(ααinvcKI::MatF64, cK::AbstractPDMat, α::Vector)
    nobs = length(α)
    size(ααinvcKI) == (nobs, nobs) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         nobs, nobs,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    fill!(ααinvcKI, 0)
    @inbounds for i in 1:nobs
        ααinvcKI[i,i] = -1.0
    end
    ldiv!(cK.chol, ααinvcKI)
    BLAS.ger!(1.0, α, α, ααinvcKI)
end

#———————————————————————————————————————————————————————————————-
#Functions for calculating the log-target

"""
    initialise_mll!(gp::GPE)

Initialise the marginal log-likelihood of Gaussian process `gp`.
"""
function initialise_mll!(gp::GPE{X,Y,M,K,P,D}) where {X,Y,M,K,P,D}
    μ = mean(gp.mean,gp.x)
    Σ = cov(gp.kernel, gp.x, gp.data)
    gp.cK = P.name.wrapper(Σ + (exp(2*gp.logNoise) + 1e-5)*I)
    y = gp.y - μ
    gp.alpha = gp.cK \ y
    # Marginal log-likelihood
    gp.mll = -(dot(y, gp.alpha) + logdet(gp.cK) + log2π * gp.nobs) / 2
    gp
end

"""
    update_cK!(gp::GPE)

Update the covariance matrix and its Cholesky decomposition of Gaussian process `gp`.
"""
function update_cK!(gp::GPE{X,Y,M,K,P,D}) where {X,Y,M,K,P,D}
    old_cK = gp.cK
    if P <: ElasticPDMat
        Σbuffer = view(old_cK.mat)
    else
        Σbuffer = old_cK.mat
    end
    cov!(Σbuffer, gp.kernel, gp.x, gp.data)
    noise = (exp(2*gp.logNoise) + 1e-5)
    for i in 1:gp.nobs
        Σbuffer[i,i] += noise
    end
    if P <: ElasticPDMat
        chol_buffer = view(old_cK.chol).factors
    else
        chol_buffer = old_cK.chol.factors
    end
    copyto!(chol_buffer, Σbuffer)
    chol = cholesky!(Symmetric(chol_buffer))
    if !(P <: ElasticPDMat)
        gp.cK = P.name.wrapper(Σbuffer, chol)
    end
    gp.cK
end

is_data_updated(gp::GPE{X,Y,M,K,P,D}) where {X,Y,M,K,P <: ElasticPDMat,D} = false

# modification of initialise_target! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_mll!(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    if kern | noise
        update_cK!(gp)
    end
    μ = mean(gp.mean, gp.x)
    y = gp.y - μ
    gp.alpha = gp.cK \ y
    # Marginal log-likelihood
    gp.mll = - (dot(y, gp.alpha) + logdet(gp.cK) + log2π * gp.nobs) / 2
    gp
end

function dmll_kern!(dmll::VecF64, k::Kernel, X::MatF64, data::KernelData, ααinvcKI::MatF64)
    dim, nobs = size(X)
    nparams = num_params(k)
    @assert nparams == length(dmll)
    dK_buffer = Vector{Float64}(undef, nparams)
    dmll[:] .= 0.0
    @inbounds for j in 1:nobs
        # off-diagonal
        for i in 1:j-1
            dKij_dθ!(dK_buffer, k, X, data, i, j, dim, nparams)
            @simd for iparam in 1:nparams
                dmll[iparam] += dK_buffer[iparam] * ααinvcKI[i, j]
            end
        end
        # diagonal
        dKij_dθ!(dK_buffer, k, X, data, j, j, dim, nparams)
        for iparam in 1:nparams
            dmll[iparam] += dK_buffer[iparam] * ααinvcKI[j, j] / 2.0
        end
    end
    return dmll
end
"""
     update_dmll!(gp::GPE, ...)

Update the gradient of the marginal log-likelihood of Gaussian process `gp`.
"""
function update_dmll!(gp::GPE, ααinvcKI::MatF64;
    noise::Bool=true, # include gradient component for the logNoise term
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    size(ααinvcKI) == (gp.nobs, gp.nobs) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         gp.nobs, gp.nobs,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)
    gp.dmll = Array{Float64}(undef, noise + domean * n_mean_params + kern * n_kern_params)

    get_ααinvcKI!(ααinvcKI, gp.cK, gp.alpha)

    i=1
    if noise
        gp.dmll[i] = exp(2 * gp.logNoise) * tr(ααinvcKI)
        i += 1
    end

    if domean && n_mean_params>0
        Mgrads = grad_stack(gp.mean, gp.x)
        for j in 1:n_mean_params
            gp.dmll[i] = dot(Mgrads[:,j], gp.alpha)
            i += 1
        end
    end
    if kern
        dmll_k = @view(gp.dmll[i:end])
        dmll_kern!(dmll_k, gp.kernel, gp.x, gp.data, ααinvcKI)
    end
end

"""
    update_mll_and_dmll!(gp::GPE, ...)

Update the gradient of the marginal log-likelihood of a Gaussian
process `gp`.
"""
function update_mll_and_dmll!(gp::GPE,
        ααinvcKI::MatF64;
        kwargs...)
    update_mll!(gp; kwargs...)
    update_dmll!(gp, ααinvcKI; kwargs...)
end

"""
    initialise_target!(gp::GPE)

Initialise the log-posterior
```math
\\log p(θ | y) ∝ \\log p(y | θ) +  \\log p(θ)
```
of a Gaussian process `gp`.
"""
function initialise_target!(gp::GPE)
    initialise_mll!(gp)
        #HOW TO SET-UP A PRIOR FOR THE LOGNOISE?
    gp.target = gp.mll   + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel)
    gp
end

"""
    update_target!(gp::GPE, ...)

Update the log-posterior
```math
\\log p(θ | y) ∝ \\log p(y | θ) +  \\log p(θ)
```
of a Gaussian process `gp`.
"""
function update_target!(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    update_mll!(gp; noise=noise, domean=domean, kern=kern)
    #HOW TO SET-UP A PRIOR FOR THE LOGNOISE?
    gp.target = gp.mll  + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel)
    gp
end

function update_dtarget!(gp::GPE, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_dmll!(gp, L_bar; kwargs...)
    gp.dtarget = gp.dmll + prior_gradlogpdf(gp; kwargs...)
    gp
end

"""
    update_target_and_dtarget!(gp::GPE, ...)

Update the log-posterior
```math
\\log p(θ | y) ∝ \\log p(y | θ) +  \\log p(θ)
```
of a Gaussian process `gp` and its derivative.
"""
function update_target_and_dtarget!(gp::GPE, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_target!(gp; kwargs...)
    update_dtarget!(gp, Kgrad, L_bar; kwargs...)
end

function update_target_and_dtarget!(gp::GPE; kwargs...)
    ααinvcKI = Array{Float64}(undef, gp.nobs, gp.nobs)
    update_target!(gp; kwargs...)
    update_dmll!(gp, ααinvcKI; kwargs...)
    gp.dtarget = gp.dmll + prior_gradlogpdf(gp; kwargs...)
end


#——————————————————————#
# Predict observations #
#——————————————————————#

"""
    predict_y(gp::GPE, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])

Return the predictive mean and variance of Gaussian Process `gp` at specfic points which
are given as columns of matrix `x`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_y(gp::GPE, x::MatF64; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    if full_cov
        return μ, σ2 + ScalMat(gp.nobs, exp(2*gp.logNoise))
    else
        return μ, σ2 .+ exp(2*gp.logNoise)
    end
end

# 1D Case for predictions
predict_y(gp::GPE, x::VecF64; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)

@deprecate predict predict_y

## compute predictions
function _predict(gp::GPE, x::MatF64)
    cK = cov(gp.kernel, gp.x, x)
    Lck = whiten(gp.cK, cK)
    mu = mean(gp.mean, x) + cK'*gp.alpha        # Predictive mean
    Sigma_raw = cov(gp.kernel, x) - Lck'Lck # Predictive covariance
    # Add jitter to get stable covariance
    Sigma = tolerant_PDMat(Sigma_raw)
    return mu, Sigma
end

#———————————————————————————————————————————————————————————
# Sample from the GPE
function Random.rand!(gp::GPE, x::MatF64, A::DenseMatrix)
    nobs = size(x,2)
    n_sample = size(A,2)

    if gp.nobs == 0
        # Prior mean and covariance
        μ = mean(gp.mean, x);
        Σraw = cov(gp.kernel, x);
        Σ = tolerant_PDMat(Σraw)
    else
        # Posterior mean and covariance
        μ, Σ = predict_f(gp, x; full_cov=true)
    end
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobs, n_sample)))
end

Random.rand(gp::GPE, x::MatF64, n::Int) = rand!(gp, x, Array{Float64}(undef, size(x, 2), n))

# Sample from 1D GPE
Random.rand(gp::GPE, x::VecF64, n::Int) = rand(gp, x', n)

# Generate only one sample from the GPE and returns a vector
Random.rand(gp::GPE, x::MatF64) = vec(rand(gp,x,1))
Random.rand(gp::GPE, x::VecF64) = vec(rand(gp,x',1))

#—————————————————————————————————————————————————————–
#Functions for setting and calling the parameters of the GP object

function get_params(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gp.logNoise); end
    if domean
        append!(params, get_params(gp.mean))
    end
    if kern
        append!(params, get_params(gp.kernel))
    end
    return params
end

function num_params(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n = 0
    noise && (n += 1)
    domean && (n += num_params(gp.mean))
    kern && (n += num_params(gp.kernelern))
    n
end

function set_params!(gp::GPE, hyp::VecF64;
                     noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)

    i = 1
    if noise
        gp.logNoise = hyp[1];
        i+=1
    end

    if domean && n_mean_params>0
        set_params!(gp.mean, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end
    if kern
        set_params!(gp.kernel, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
end

function prior_gradlogpdf(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    grad = Float64[]
    if noise; push!(grad, 0.0); end # Noise does not have any priors
    if domean
        append!(grad, prior_gradlogpdf(gp.mean))
    end
    if kern
        append!(grad, prior_gradlogpdf(gp.kernel))
    end
    return grad
end

#———————————————————————————————————————————————————————————-
# Push function
function Base.push!(gp::GPE, x::MatF64, y::VecF64)
    @warn "push! method is currently inefficient as it refits all observations"
    if gp.nobs == 0
        GaussianProcesses.fit!(gp, x, y)
    elseif size(x,1) != size(gp.x,1)
        error("New input observations must have dimensions consistent with existing observations")
    else
        GaussianProcesses.fit!(gp, cat(2, gp.x, x), cat(1, gp.y, y))
    end
end

Base.push!(gp::GPE, x::VecF64, y::VecF64) = push!(gp, x', y)
Base.push!(gp::GPE, x::Float64, y::Float64) = push!(gp, [x], [y])
Base.push!(gp::GPE, x::VecF64, y::Float64) = push!(gp, reshape(x, length(x), 1), [y])


# —————————————————————————————————————————————————————————————
# Show function
function Base.show(io::IO, gp::GPE)
    println(io, "GP Exact object:")
    println(io, "  Dim = ", gp.dim)
    println(io, "  Number of observations = ", gp.nobs)
    println(io, "  Mean function:")
    show(io, gp.mean, 2)
    println(io, "\n  Kernel:")
    show(io, gp.kernel, 2)
    if gp.nobs == 0
        println("\n  No observation data")
    else
        println(io, "\n  Input observations = ")
        show(io, gp.x)
        print(io, "\n  Output observations = ")
        show(io, gp.y)
        print(io, "\n  Variance of observation noise = ", exp(2 * gp.logNoise))
        print(io, "\n  Marginal Log-Likelihood = ")
        show(io, round(gp.target; digits = 3))
    end
end
