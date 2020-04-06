# Main GaussianProcess type

mutable struct GPE{X<:AbstractMatrix,Y<:AbstractVector,M<:Mean,K<:Kernel,CS<:CovarianceStrategy,D<:KernelData,P<:AbstractPDMat,NOI<:Param} <: GPBase
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
    logNoise::NOI
    "Strategy for computing or approximating covariance matrices"
    covstrat::CS

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

    function GPE{X,Y,M,K,CS,D,P,NOI}(x::X, y::Y, mean::M, kernel::K, logNoise::NOI, covstrat::CS, kerneldata::D, cK::P) where {X,Y,M,K,CS,D,P,NOI}
        dim, nobs = size(x)
        length(y) == nobs || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new{X,Y,M,K,CS,D,P,NOI}(x, y, mean, kernel, logNoise, covstrat, dim, nobs, kerneldata, cK)
        initialise_target!(gp)
    end
end

function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Param, covstrat::CovarianceStrategy, kerneldata::KernelData, cK::AbstractPDMat)
    GPE{typeof(x),typeof(y),typeof(mean),typeof(kernel),typeof(covstrat),typeof(kerneldata),typeof(cK), typeof(logNoise)}(x, y, mean, kernel, logNoise, covstrat,kerneldata, cK)
end
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Union{Real,AbstractVector}, covstrat::CovarianceStrategy, kerneldata::KernelData, cK::AbstractPDMat)
    lns = wrap_param(logNoise)
    gp = GPE(x, y, mean, kernel, lns, covstrat, kerneldata, cK)
    return gp
end
function alloc_cK(nobs)
    # create placeholder PDMat
    m = Matrix{Float64}(undef, nobs, nobs)
    chol = Matrix{Float64}(undef, nobs, nobs)
    cK = PDMats.PDMat(m, Cholesky(chol, 'U', 0))
    return cK
end
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise, covstrat::CovarianceStrategy, kerneldata::KernelData)
    nobs = length(y)
    cK = alloc_cK(covstrat, nobs)
    GPE(x, y, mean, kernel, logNoise, covstrat, kerneldata, cK)
end
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise, covstrat::CovarianceStrategy)
    kerneldata = KernelData(kernel, x, x, covstrat)
    GPE(x, y, mean, kernel, logNoise, covstrat, kerneldata)
end
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise, kerneldata::KernelData)
    covstrat = FullCovariance()
    GPE(x, y, mean, kernel, logNoise, covstrat, kerneldata)
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
- `logNoise`: Natural logarithm of the standard deviation for the observation
  noise. The default is -2.0, which is equivalent to assuming no observation noise.
  Can be a vector for heteroscedastic noise.
"""
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise=-2.0)
    covstrat = FullCovariance()
    GPE(x, y, mean, kernel, logNoise, covstrat)
end
GPE(x::AbstractVector, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise = -2.0) =
    GPE(x', y, mean, kernel, logNoise)

"""
    GPE(; mean::Mean = MeanZero(), kernel::Kernel = SE(0.0, 0.0), logNoise = -2.0)

Construct a [`GPE`](@ref) object without observations.
"""
function GPE(; mean::Mean = MeanZero(), kernel::Kernel = SE(0.0, 0.0), logNoise = -2.0)
    x = Array{Float64}(undef, 1, 0) # ElasticArrays don't like length(x) = 0.
    y = Array{Float64}(undef, 0)
    GPE(x, y, mean, kernel, logNoise)
end

"""
    GP(x, y, mean::Mean, kernel::Kernel[, logNoise=-2.0])

Fit a Gaussian process that is defined by its `mean`, its `kernel`, and the logarithm
`logNoise` of the standard deviation of its observation noise to a set of training points
`x` and `y`.

See also: [`GPE`](@ref)
"""
GP(x::AbstractVecOrMat{Float64}, y::AbstractVector, mean::Mean, kernel::Kernel,
   logNoise = -2.0) = GPE(x, y, mean, kernel, logNoise)

"""
    fit!(gp::GPE{X,Y}, x::X, y::Y)

Fit Gaussian process `GPE` to a training data set consisting of input observations `x` and
output observations `y`.
"""
function fit!(gp::GPE{X,Y}, x::X, y::Y) where {X,Y}
    length(y) == size(x,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.x = x
    gp.y = y
    gp.data = KernelData(gp.kernel, x, x, gp.covstrat)
    gp.cK = alloc_cK(gp.covstrat, length(y))
    gp.dim, gp.nobs = size(x)
    initialise_target!(gp)
end

fit!(gp::GPE, x::AbstractVector, y::AbstractVector) = fit!(gp, x', y)

#———————————————————————————————————————————————————————————
#Fast memory allocation function

LinearAlgebra.ldiv!(cK::PDMat, x) = ldiv!(cK.chol, x)
"""
    get_ααinvcKI!(ααinvcKI::Matrix{Float64}, cK::AbstractPDMat, α::Vector)

Write `ααᵀ - cK⁻¹` to `ααinvcKI` avoiding any memory allocation, where `cK` and
`α` are the covariance matrix and the alpha vector of a Gaussian process, respectively.
Hereby `α` is defined as `cK⁻¹ (Y - μ)`.
"""
function get_ααinvcKI!(ααinvcKI::AbstractMatrix, cK::AbstractPDMat, α::Vector)
    nobs = length(α)
    size(ααinvcKI) == (nobs, nobs) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         nobs, nobs,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    fill!(ααinvcKI, 0)
    @inbounds for i in 1:nobs
        ααinvcKI[i,i] = -1.0
    end
    # `ldiv!(A, B)`: Compute A \ B in-place and overwriting B to store the result.
    ldiv!(cK, ααinvcKI)
    BLAS.ger!(1.0, α, α, ααinvcKI)
end

#———————————————————————————————————————————————————————————————-
#Functions for calculating the log-target

function update_cK!(cK::AbstractPDMat, x::AbstractMatrix, kernel::Kernel, logNoise::Real, data::KernelData, covstrat::CovarianceStrategy)
    nobs = size(x, 2)
    Σbuffer = mat(cK)
    cov!(Σbuffer, kernel, x, x, data)
    noise = exp(2*logNoise)+eps()
    for i in 1:nobs
        Σbuffer[i,i] += noise
    end
    Σbuffer, chol = make_posdef!(Σbuffer, cholfactors(cK))
    return wrap_cK(cK, Σbuffer, chol)
end
function update_cK!(cK::AbstractPDMat, x::AbstractMatrix, kernel::Kernel, logNoise::AbstractVector, data::KernelData, covstrat::CovarianceStrategy)
    nobs = size(x, 2)
    Σbuffer = mat(cK)
    cov!(Σbuffer, kernel, x, x, data)
    noise = exp.(2 .* logNoise) .+ eps()
    for i in 1:nobs
        Σbuffer[i,i] += noise[i]
    end
    Σbuffer, chol = make_posdef!(Σbuffer, cholfactors(cK))
    return wrap_cK(cK, Σbuffer, chol)
end

"""
    update_cK!(gp::GPE)

Update the covariance matrix and its Cholesky decomposition of Gaussian process `gp`.
"""
function update_cK!(gp::GPE)
    gp.cK = update_cK!(gp.cK, gp.x, gp.kernel, get_value(gp.logNoise), gp.data, gp.covstrat)
end

"""
    update_mll!(gp::GPE)
    
Modification of initialise_target! that reuses existing matrices to avoid unnecessary memory allocations, which speeds things up significantly.
"""    
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

"""
    dmll_kern!((dmll::AbstractVector, k::Kernel, X::AbstractMatrix, data::KernelData, ααinvcKI::AbstractMatrix))

Derivative of the marginal log likelihood log p(Y|θ) with respect to the kernel hyperparameters.
"""
function dmll_kern!(dmll::AbstractVector, k::Kernel, X::AbstractMatrix, data::KernelData, 
                    ααinvcKI::Matrix{Float64}, covstrat::CovarianceStrategy)
    dim, nobs = size(X)
    nparams = num_params(k)
    @assert nparams == length(dmll)
    dK_buffer = Vector{Float64}(undef, nparams)
    dmll[:] .= 0.0
    @inbounds for j in 1:nobs
        # diagonal
        dKij_dθ!(dK_buffer, k, X, X, data, j, j, dim, nparams)
        for iparam in 1:nparams
            dmll[iparam] += dK_buffer[iparam] * ααinvcKI[j, j] / 2.0
        end
        # off-diagonal
        for i in j+1:nobs
            dKij_dθ!(dK_buffer, k, X, X, data, i, j, dim, nparams)
            @simd for iparam in 1:nparams
                dmll[iparam] += dK_buffer[iparam] * ααinvcKI[i, j]
            end
        end
    end
    return dmll
end

""" AbstractGradientPrecompute types hold results of
    pre-computations of kernel gradients.
"""
abstract type AbstractGradientPrecompute end

struct FullCovariancePrecompute <: AbstractGradientPrecompute
    ααinvcKI::Matrix{Float64}
end
function FullCovariancePrecompute(nobs::Int)
    buffer = Matrix{Float64}(undef, nobs, nobs)
    return FullCovariancePrecompute(buffer)
end

function init_precompute(covstrat::FullCovariance, X, y, k)
    nobs = size(X, 2)
    FullCovariancePrecompute(nobs)
end
init_precompute(gp::GPBase) = init_precompute(gp.covstrat, gp.x, gp.y, gp.kernel)
    
function precompute!(precomp::FullCovariancePrecompute, gp::GPBase) 
    get_ααinvcKI!(precomp.ααinvcKI, gp.cK, gp.alpha)
end
function dmll_kern!(dmll::AbstractVector, gp::GPBase, precomp::FullCovariancePrecompute, covstrat::CovarianceStrategy)
    return dmll_kern!(dmll, gp.kernel, gp.x, gp.data, precomp.ααinvcKI, covstrat)
end

noise_variance(gp::GPE) = noise_variance(gp.logNoise)
noise_variance(logNoise::Scalar) = exp(2 * get_value(logNoise))
noise_variance(logNoise::VectorParam) = exp.(2 .* get_value(logNoise))

function dmll_noise(logNoise::Real, precomp::FullCovariancePrecompute)
    return exp(2 * logNoise) * tr(precomp.ααinvcKI)
end
# function dmll_noise(logNoise::AbstractVector, precomp::FullCovariancePrecompute)
    # return exp.(2 .* logNoise) .* diag(precomp.ααinvcKI)
# end
function dmll_noise(gp::GPE, precomp::FullCovariancePrecompute, covstrat::CovarianceStrategy)
    dmll_noise(get_value(gp.logNoise), precomp)
end
function dmll_mean!(dmll::AbstractVector, meanf::Mean, x::AbstractMatrix, alpha::AbstractVector)
    Mgrads = grad_stack(meanf, x)
    for j in 1:num_params(meanf)
        dmll[j] = dot(Mgrads[:,j], alpha)
    end
    return dmll
end
function dmll_mean!(dmll::AbstractVector, gp::GPBase, precomp::AbstractGradientPrecompute)
    dmll_mean!(dmll, gp.mean, gp.x, gp.alpha)
end

"""
     update_dmll!(gp::GPE, ...)

Update the gradient of the marginal log-likelihood of Gaussian process `gp`.
"""
function update_dmll!(gp::GPE, precomp::AbstractGradientPrecompute;
    noise::Bool=true, # include gradient component for the logNoise term
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)
    gp.dmll = Array{Float64}(undef, noise + domean * n_mean_params + kern * n_kern_params)
    precompute!(precomp, gp)

    i=1
    if noise
        @assert num_params(gp.logNoise) == 1
        gp.dmll[i] = dmll_noise(gp, precomp, gp.covstrat)
        i += 1
    end

    if domean && n_mean_params>0
        dmll_m = @view(gp.dmll[i:i+n_mean_params-1])
        dmll_mean!(dmll_m, gp, precomp)
        i += n_mean_params
    end
    if kern
        dmll_k = @view(gp.dmll[i:end])
        dmll_kern!(dmll_k, gp, precomp, gp.covstrat)
    end
end

"""
    update_mll_and_dmll!(gp::GPE, ...)

Update the gradient of the marginal log-likelihood of a Gaussian
process `gp`.
"""
function update_mll_and_dmll!(gp::GPE, precomp::AbstractGradientPrecompute; kwargs...)
    update_mll!(gp; kwargs...)
    update_dmll!(gp, precomp; kwargs...)
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
    update_mll!(gp)
    gp.target = gp.mll + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel) + prior_logpdf(gp.logNoise)
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
    gp.target = gp.mll  + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel) + prior_logpdf(gp.logNoise)
    gp
end

function update_dtarget!(gp::GPE, precomp::AbstractGradientPrecompute; kwargs...)
    update_dmll!(gp, precomp; kwargs...)
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
function update_target_and_dtarget!(gp::GPE, precomp::AbstractGradientPrecompute; kwargs...)
    update_target!(gp; kwargs...)
    update_dtarget!(gp, precomp; kwargs...)
end

function update_target_and_dtarget!(gp::GPE; kwargs...)
    precomp = init_precompute(gp)
    update_target!(gp; kwargs...)
    update_dmll!(gp, precomp; kwargs...)
    gp.dtarget = gp.dmll + prior_gradlogpdf(gp; kwargs...)
end


#——————————————————————#
# Predict observations #
#——————————————————————#

predict_full(gp::GPE, xpred::AbstractMatrix) = predictMVN(xpred, gp.x, gp.y, gp.kernel, gp.mean, gp.alpha, gp.covstrat, gp.cK)
"""
    predict_full(gp::GPE, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])

Return the predictive mean and variance of Gaussian Process `gp` at specfic points which
are given as columns of matrix `x`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""

function predict_y(gp::GPE, x::AbstractMatrix; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    if full_cov
        npred = size(x, 2)
        return μ, σ2 + ScalMat(npred, noise_variance(gp))
    else
        return μ, σ2 .+ noise_variance(gp)
    end
end

#—————————————————————————————————————————————————————–
# Function for sampling from the prior of the GP object hyperparameters.

function sample_params(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    samples = Float64[]
    if noise && num_params(gp.logNoise) != 0
        noise_priors = get_priors(gp.logNoise)
        @assert !isempty(noise_priors) "prior distributions of logNoise should be set"
        noise_sample = rand(Product(noise_priors))
        append!(samples, noise_sample)
    end
    if domean && num_params(gp.mean) != 0
        mean_priors = get_priors(gp.mean)
        @assert !isempty(mean_priors) "prior distributions of mean should be set"
        mean_sample = rand(Product(mean_priors))
        append!(samples, mean_sample)
    end
    if kern && num_params(gp.kernel) != 0
        kernel_priors = get_priors(gp.kernel)
        @assert !isempty(kernel_priors) "prior distributions of kernel should be set"
        kernel_sample = rand(Product(kernel_priors))
        append!(samples, kernel_sample)
    end
    return samples
end

#—————————————————————————————————————————————————————–
#Functions for setting and calling the parameters of the GP object

function get_params(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; append!(params, get_params(gp.logNoise)); end
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
    noise && (n += num_params(gp.logNoise))
    domean && (n += num_params(gp.mean))
    kern && (n += num_params(gp.kernel))
    n
end

function appendbounds!(lb, ub, n, bounds)
    n == 0 && return
    if bounds == nothing
        append!(lb, fill(-Inf, n))
        append!(ub, fill(Inf, n))
    else
        append!(lb, bounds[1])
        append!(ub, bounds[2])
    end
    return
end
appendnoisebounds!(lb, ub, gp::GPE, bounds) = appendbounds!(lb, ub, 1, bounds)
appendnoisebounds!(lb, ub, gp, bounds) = Nothing
appendlikbounds!(lb, ub, gp, bounds) = Nothing
function bounds(gp::GPBase, noisebounds, meanbounds, kernbounds, likbounds;
                noise::Bool=true, domean::Bool=true, kern::Bool=true, lik::Bool=true)
    lb = Float64[]
    ub = Float64[]
    noise && appendnoisebounds!(lb, ub, gp, noisebounds)
    lik && appendlikbounds!(lb, ub, gp, likbounds)
    domean && appendbounds!(lb, ub, num_params(gp.mean), meanbounds)
    kern && appendbounds!(lb, ub, num_params(gp.kernel), kernbounds)
    lb, ub
end

function set_params!(gp::GPE, hyp::AbstractVector;
                     noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n_noise_params = num_params(gp.logNoise)
    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)

    i = 1
    if noise
        set_params!(gp.logNoise, hyp[1:n_noise_params])
        i += n_noise_params
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
    if noise
        append!(grad, prior_gradlogpdf(gp.logNoise))
    end
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
function Base.push!(gp::GPE, x::AbstractMatrix, y::AbstractVector)
    @warn "push! method is currently inefficient as it refits all observations"
    if gp.nobs == 0
        GaussianProcesses.fit!(gp, x, y)
    elseif size(x,1) != size(gp.x,1)
        error("New input observations must have dimensions consistent with existing observations")
    else
        GaussianProcesses.fit!(gp, cat(2, gp.x, x), cat(1, gp.y, y))
    end
end

Base.push!(gp::GPE, x::AbstractVector, y::AbstractVector) = push!(gp, x', y)
Base.push!(gp::GPE, x::Float64, y::Float64) = push!(gp, [x], [y])
Base.push!(gp::GPE, x::AbstractVector, y::Float64) = push!(gp, reshape(x, length(x), 1), [y])


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
        print(io, "\n  Variance of observation noise = ", noise_variance(gp))
        print(io, "\n  Marginal Log-Likelihood = ")
        show(io, round(gp.target; digits = 3))
    end
end
