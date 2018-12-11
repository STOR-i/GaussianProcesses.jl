# Main GaussianProcess type

mutable struct GPE{X<:AbstractMatrix,Y<:AbstractVector,M<:Mean,K<:Kernel,P<:AbstractPDMat,D<:KernelData} <: GPBase
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

    function GPE{X,Y,M,K,P,D}(x::X, y::Y, mean::M, kernel::K, logNoise::Float64, data::D, cK::P) where {X,Y,M,K,P,D}
        dim, nobs = size(x)
        length(y) == nobs || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new{X,Y,M,K,P,D}(x, y, mean, kernel, logNoise, dim, nobs, data, cK)
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
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Float64, kerneldata::KernelData, cK::AbstractPDMat) 
    GPE{typeof(x),typeof(y),typeof(mean),typeof(kernel),typeof(cK),typeof(kerneldata)}(x, y, mean, kernel, logNoise, kerneldata, cK)
end
function alloc_cK(nobs)
    # create placeholder PDMat
    m = Matrix{Float64}(undef, nobs, nobs)
    chol = Matrix{Float64}(undef, nobs, nobs)
    cK = PDMats.PDMat(m, Cholesky(chol, 'U', 0))
    return cK
end
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Float64, kerneldata::KernelData)
    nobs = length(y)
    cK = alloc_cK(nobs)
    GPE(x, y, mean, kernel, logNoise, kerneldata, cK)
end
function GPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Float64 = -2.0)
    kerneldata = KernelData(kernel, x, x)
    GPE(x, y, mean, kernel, logNoise, kerneldata)
end
GPE(x::AbstractVector, y::AbstractVector, mean::Mean, kernel::Kernel, logNoise::Float64 = -2.0) =
    GPE(x', y, mean, kernel, logNoise)

"""
    GPE(; mean::Mean = MeanZero(), kernel::Kernel = SE(0.0, 0.0), logNoise::Float64 = -2.0)

Construct a [`GPE`](@ref) object without observations.
"""
function GPE(; mean::Mean = MeanZero(), kernel::Kernel = SE(0.0, 0.0), logNoise::Float64 = -2.0) 
    x = Array{Float64}(undef, 1, 0) # ElasticArrays don't like length(x) = 0.
    y = Array{Float64}(undef, 0)
    GPE(x, y, mean, kernel, logNoise)
end

"""
    GP(x, y, mean::Mean, kernel::Kernel[, logNoise::Float64=-2.0])

Fit a Gaussian process that is defined by its `mean`, its `kernel`, and the logarithm
`logNoise` of the standard deviation of its observation noise to a set of training points
`x` and `y`.

See also: [`GPE`](@ref)
"""
GP(x::AbstractVecOrMat{Float64}, y::AbstractVector, mean::Mean, kernel::Kernel,
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
    gp.data = KernelData(gp.kernel, x, x)
    gp.cK = alloc_cK(length(y))
    gp.dim, gp.nobs = size(x)
    initialise_target!(gp)
end

fit!(gp::GPE, x::AbstractVector, y::AbstractVector) = fit!(gp, x', y)

#———————————————————————————————————————————————————————————
#Fast memory allocation function

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
    ldiv!(cK.chol, ααinvcKI)
    BLAS.ger!(1.0, α, α, ααinvcKI)
end

#———————————————————————————————————————————————————————————————-
#Functions for calculating the log-target

"""
    update_cK!(gp::GPE)

Update the covariance matrix and its Cholesky decomposition of Gaussian process `gp`.
"""
function update_cK!(gp::GPE)
    old_cK = gp.cK
    Σbuffer = mat(old_cK)
    cov!(Σbuffer, gp.kernel, gp.x, gp.data)
    noise = exp(2*gp.logNoise)+eps()
    for i in 1:gp.nobs
        Σbuffer[i,i] += noise
    end
    Σbuffer, chol = make_posdef!(Σbuffer, cholfactors(old_cK))
    gp.cK = wrap_cK(gp.cK, Σbuffer, chol)
    # copyto!(chol_buffer, Σbuffer)
    # chol = cholesky!(Symmetric(chol_buffer))
    # gp.cK = wrap_cK(gp.cK, Σbuffer, chol)
    # gp.cK = new_cK
end

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

function dmll_kern!(dmll::AbstractVector, k::Kernel, X::AbstractMatrix, data::KernelData, ααinvcKI::AbstractMatrix)
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
function update_dmll!(gp::GPE, ααinvcKI::AbstractMatrix;
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
        ααinvcKI::AbstractMatrix;
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
    update_mll!(gp)
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

function update_dtarget!(gp::GPE, Kgrad::AbstractMatrix, L_bar::AbstractMatrix; kwargs...)
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
function update_target_and_dtarget!(gp::GPE, Kgrad::AbstractMatrix, L_bar::AbstractMatrix; kwargs...)
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
function predict_y(gp::GPE, x::AbstractMatrix; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    if full_cov
        return μ, σ2 + ScalMat(gp.nobs, exp(2*gp.logNoise))
    else
        return μ, σ2 .+ exp(2*gp.logNoise)
    end
end

# 1D Case for predictions
predict_y(gp::GPE, x::AbstractVector; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)

@deprecate predict predict_y

## compute predictions
function _predict(gp::GPE, x::AbstractMatrix)
    crossdata = KernelData(gp.kernel, gp.x, x)
    priordata = KernelData(gp.kernel, x, x)
    cK = cov(gp.kernel, gp.x, x, crossdata)
    mu = mean(gp.mean, x) + cK'*gp.alpha        # Predictive mean
    Lck = whiten!(gp.cK, cK)
    Sigma_raw = cov(gp.kernel, x, x, priordata)
    subtract_Lck!(Sigma_raw, Lck)
    # Add jitter to get stable covariance
    m, chol = make_posdef!(Sigma_raw)
    return mu, PDMat(m, chol)
end
@inline function subtract_Lck!(Sigma_raw::AbstractArray{<:AbstractFloat}, Lck::AbstractArray{<:AbstractFloat})
    LinearAlgebra.BLAS.syrk!('U', 'T', -1.0, Lck, 1.0, Sigma_raw)
    LinearAlgebra.copytri!(Sigma_raw, 'U')
end
@inline subtract_Lck!(Sigma_raw, Lck) = Sigma_raw .-= Lck'Lck


#———————————————————————————————————————————————————————————
# Sample from the GPE
function Random.rand!(gp::GPE, x::AbstractMatrix, A::DenseMatrix)
    nobs = size(x,2)
    n_sample = size(A,2)

    if gp.nobs == 0
        # Prior mean and covariance
        μ = mean(gp.mean, x);
        Σraw = cov(gp.kernel, x);
        Σraw, chol = make_posdef!(Σraw)
        Σ = PDMat(Σraw, chol)
    else
        # Posterior mean and covariance
        μ, Σ = predict_f(gp, x; full_cov=true)
    end
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobs, n_sample)))
end

Random.rand(gp::GPE, x::AbstractMatrix, n::Int) = rand!(gp, x, Array{Float64}(undef, size(x, 2), n))

# Sample from 1D GPE
Random.rand(gp::GPE, x::AbstractVector, n::Int) = rand(gp, x', n)

# Generate only one sample from the GPE and returns a vector
Random.rand(gp::GPE, x::AbstractMatrix) = vec(rand(gp,x,1))
Random.rand(gp::GPE, x::AbstractVector) = vec(rand(gp,x',1))

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
        print(io, "\n  Variance of observation noise = ", exp(2 * gp.logNoise))
        print(io, "\n  Marginal Log-Likelihood = ")
        show(io, round(gp.target; digits = 3))
    end
end
