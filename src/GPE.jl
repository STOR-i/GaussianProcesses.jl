# Main GaussianProcess type

mutable struct GPE <: GPBase
    "Mean object"
    m:: Mean
    "Kernel object"
    k::Kernel
    "Log standard deviation of observation noise"
    logNoise::Float64

    # Observation data
    "Number of observations"
    nobsv::Int
    "Input observations"
    X::MatF64
    "Output observations"
    y::VecF64
    "Auxiliary observation data (to speed up calculations)"
    data::KernelData
    "Dimension of inputs"
    dim::Int

    # Auxiliary data
    "`(k + exp(2*obsNoise))`"
    cK::AbstractPDMat
    "`(k + exp(2*obsNoise))⁻¹y`"
    alpha::VecF64
    "Marginal log-likelihood"
    mll::Float64
    "Log target (marginal log-likelihood + log priors)"
    target::Float64
    "Gradient of marginal log-likelihood"
    dmll::VecF64
    "Gradient of log-target (gradient of marginal log-likelihood + gradient of log priors)"
    dtarget::VecF64

    """
        GPE(X, y, m, k, logNoise)
        GPE(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-2.0) # observation-free constructor

    Fit a Gaussian process to a set of training points. The Gaussian process is defined in
    terms of its user-defined mean and covariance (kernel) functions. As a default it is
    assumed that the observations are noise free.

    # Arguments:
    - `X::Matrix{Float64}`: Input observations
    - `y::Vector{Float64}`: Output observations
    - `m::Mean`: Mean function
    - `k::Kernel`: Covariance function
    - `logNoise::Float64`: Natural logarithm of the standard deviation for the observation
      noise. The default is -2.0, which is equivalent to assuming no observation noise.
    """
    function GPE(X::MatF64, y::VecF64, m::Mean, k::Kernel, logNoise::Float64=-2.0)
        dim, nobsv = size(X)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(m, k, logNoise, nobsv, X, y, KernelData(k, X), dim)
        initialise_target!(gp)
        return gp
    end

    GPE(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-2.0) =  new(m, k, logNoise, 0)
    GPE(m::Mean, k::Kernel, logNoise::Float64, nobsv::Int,
        X::MatF64, y::VecF64,
        data::KernelData, dim::Int,
        cK::AbstractPDMat, alpha::VecF64,
        mll::Float64, target::Float64,
        dmll::VecF64, dtarget::VecF64) = new(
                m, k, logNoise, nobsv,
                X, y, data, dim, cK, alpha,
                mll, target, dmll, dtarget)
end

GP(X::MatF64, y::VecF64, m::Mean, k::Kernel, logNoise::Float64=-2.0) = GPE(X, y, m, k, logNoise)

# Creates GPE object for 1D case
GPE(x::VecF64, y::VecF64, meanf::Mean, kernel::Kernel, logNoise::Float64=-2.0) = GPE(x', y, meanf, kernel, logNoise)

GP(x::VecF64, y::VecF64, m::Mean, k::Kernel, logNoise::Float64=-2.0) = GPE(x', y, m, k, logNoise)


"""
    fit!(gp::GPE, X::Matrix{Float64}, y::Vector{Float64})

Fit Gaussian process `GPE` to a training data set consisting of input observations `X` and
output observations `y`.
"""
function fit!(gp::GPE, X::MatF64, y::VecF64)
    length(y) == size(X,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.X = X
    gp.y = y
    gp.data = KernelData(gp.k, X)
    gp.dim, gp.nobsv = size(X)
    initialise_target!(gp)
    return gp
end

fit!(gp::GPE, x::VecF64, y::VecF64) = fit!(gp, x', y)

#———————————————————————————————————————————————————————————
#Fast memory allocation function

"""
    get_ααinvcKI!(ααinvcKI::Matrix{Float64}, cK::AbstractPDMat, α::Vector)

Write `ααᵀ - cK * eye(nobsv)` to `ααinvcKI` avoiding any memory allocation, where `cK` and
`α` are the covariance matrix and the alpha vector of a Gaussian process, respectively.
Hereby `α` is defined as `cK \\ (Y - μ)`.
"""
function get_ααinvcKI!(ααinvcKI::MatF64, cK::AbstractPDMat, α::Vector)
    nobsv = length(α)
    size(ααinvcKI) == (nobsv, nobsv) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         nobsv, nobsv,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    fill!(ααinvcKI, 0)
    @inbounds for i in 1:nobsv
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
function initialise_mll!(gp::GPE)
    μ = mean(gp.m,gp.X)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + (exp(2*gp.logNoise) + 1e-5)*I)
    gp.alpha = gp.cK \ (gp.y - μ)
    gp.mll = -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

"""
    update_cK!(gp::GPE)

Update the covariance matrix and its Cholesky decomposition of Gaussian process `gp`.
"""
function update_cK!(gp::GPE)
    old_cK = gp.cK
    Σbuffer = old_cK.mat
    cov!(Σbuffer, gp.k, gp.X, gp.data)
    noise = (exp(2*gp.logNoise) + 1e-5)
    for i in 1:gp.nobsv
        Σbuffer[i,i] += noise
    end
    chol_buffer = old_cK.chol.factors
    copyto!(chol_buffer, Σbuffer)
    chol = cholesky!(Symmetric(chol_buffer))
    gp.cK = PDMats.PDMat(Σbuffer, chol)
end

# modification of initialise_target! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_mll!(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    if kern | noise
        update_cK!(gp)
    end
    μ = mean(gp.m,gp.X)
    gp.alpha = gp.cK \ (gp.y - μ)
    gp.mll = -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

"""
     update_dmll!(gp::GPE, ...)

Update the gradient of the marginal log-likelihood of Gaussian process `gp`.
"""
function update_dmll!(gp::GPE, Kgrad::MatF64, ααinvcKI::MatF64;
    noise::Bool=true, # include gradient component for the logNoise term
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    size(Kgrad) == (gp.nobsv, gp.nobsv) || throw(ArgumentError(
                @sprintf("Buffer for Kgrad should be a %dx%d matrix, not %dx%d",
                         gp.nobsv, gp.nobsv,
                         size(Kgrad,1), size(Kgrad,2))))
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)
    gp.dmll = Array{Float64}(undef, noise + domean * n_mean_params + kern * n_kern_params)

    get_ααinvcKI!(ααinvcKI, gp.cK, gp.alpha)

    i=1
    if noise
        gp.dmll[i] = exp(2 * gp.logNoise) * tr(ααinvcKI)
        i += 1
    end

    if domean && n_mean_params>0
        Mgrads = grad_stack(gp.m, gp.X)
        for j in 1:n_mean_params
            gp.dmll[i] = dot(Mgrads[:,j], gp.alpha)
            i += 1
        end
    end
    if kern
        for iparam in 1:n_kern_params
            grad_slice!(Kgrad, gp.k, gp.X, gp.data, iparam)
            gp.dmll[i] = dot(Kgrad, ααinvcKI) / 2
            i += 1
        end
    end
end

"""
    update_mll_and_dmll!(gp::GPE, ...)

Update the gradient of the marginal log-likelihood of a Gaussian
process `gp`.
"""
function update_mll_and_dmll!(gp::GPE,
        Kgrad::MatF64,
        ααinvcKI::MatF64;
        kwargs...)
    update_mll!(gp; kwargs...)
    update_dmll!(gp, Kgrad, ααinvcKI; kwargs...)
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
    gp.target = gp.mll   + prior_logpdf(gp.m) + prior_logpdf(gp.k)
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
    gp.target = gp.mll  + prior_logpdf(gp.m) + prior_logpdf(gp.k)
end

function update_dtarget!(gp::GPE, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_dmll!(gp, Kgrad, L_bar; kwargs...)
    gp.dtarget = gp.dmll + prior_gradlogpdf(gp; kwargs...)
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
    Kgrad = Array{Float64}(undef, gp.nobsv, gp.nobsv)
    ααinvcKI = Array{Float64}(undef, gp.nobsv, gp.nobsv)
    update_target!(gp; kwargs...)
    update_dmll!(gp, Kgrad, ααinvcKI; kwargs...)
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
        return μ, σ2 + ScalMat(gp.nobsv, exp(2*gp.logNoise))
    else
        return μ, σ2 .+ exp(2*gp.logNoise)
    end
end

# 1D Case for predictions
predict_y(gp::GPE, x::VecF64; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)

@deprecate predict predict_y

## compute predictions
function _predict(gp::GPE, X::MatF64)
    cK = cov(gp.k, gp.X, X)
    Lck = whiten(gp.cK, cK)
    mu = mean(gp.m, X) + cK'*gp.alpha        # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Add jitter to get stable covariance
    Sigma = tolerant_PDMat(Sigma_raw)
    return mu, Sigma
end

#———————————————————————————————————————————————————————————
# Sample from the GPE
function Random.rand!(gp::GPE, X::MatF64, A::DenseMatrix)
    nobsv = size(X,2)
    n_sample = size(A,2)

    if gp.nobsv == 0
        # Prior mean and covariance
        μ = mean(gp.m, X);
        Σraw = cov(gp.k, X);
        Σ = tolerant_PDMat(Σraw)
    else
        # Posterior mean and covariance
        μ, Σ = predict_f(gp, X; full_cov=true)
    end

    unwhiten!(Σ, randn(nobsv, n_sample))
    A .= μ .+ Σ
    A
end

function Random.rand(gp::GPE, X::MatF64, n::Int)
    nobsv=size(X,2)
    A = Array{Float64}( nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GPE
Random.rand(gp::GPE, x::VecF64, n::Int) = rand(gp, x', n)

# Generate only one sample from the GPE and returns a vector
Random.rand(gp::GPE, X::MatF64) = vec(rand(gp,X,1))
Random.rand(gp::GPE, x::VecF64) = vec(rand(gp,x',1))

#—————————————————————————————————————————————————————–
#Functions for setting and calling the parameters of the GP object

function get_params(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gp.logNoise); end
    if domean
        append!(params, get_params(gp.m))
    end
    if kern
        append!(params, get_params(gp.k))
    end
    return params
end

function num_params(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n = 0
    noise && (n += 1)
    domean && (n += num_params(gp.m))
    kern && (n += num_params(gp.kern))
    n
end

function set_params!(gp::GPE, hyp::VecF64;
                     noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    i = 1
    if noise
        gp.logNoise = hyp[1];
        i+=1
    end

    if domean && n_mean_params>0
        set_params!(gp.m, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end
    if kern
        set_params!(gp.k, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
end

function prior_gradlogpdf(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    grad = Float64[]
    if noise; push!(grad, 0.0); end # Noise does not have any priors
    if domean
        append!(grad, prior_gradlogpdf(gp.m))
    end
    if kern
        append!(grad, prior_gradlogpdf(gp.k))
    end
    return grad
end

#———————————————————————————————————————————————————————————-
# Push function
function Base.push!(gp::GPE, X::MatF64, y::VecF64)
    @warn "push! method is currently inefficient as it refits all observations"
    if gp.nobsv == 0
        GaussianProcesses.fit!(gp, X, y)
    elseif size(X,1) != size(gp.X,1)
        error("New input observations must have dimensions consistent with existing observations")
    else
        GaussianProcesses.fit!(gp, cat(2, gp.X, X), cat(1, gp.y, y))
    end
end

Base.push!(gp::GPE, x::VecF64, y::VecF64) = push!(gp, x', y)
Base.push!(gp::GPE, x::Float64, y::Float64) = push!(gp, [x], [y])
Base.push!(gp::GPE, x::VecF64, y::Float64) = push!(gp, reshape(x, length(x), 1), [y])



#—————————————————————————————————————————————————————————————
#Show function
function Base.show(io::IO, gp::GPE)
    println(io, "GP Exact object:")
    println(io, "  Dim = $(gp.dim)")
    println(io, "  Number of observations = $(gp.nobsv)")
    println(io, "  Mean function:")
    show(io, gp.m, 2)
    println(io, "  Kernel:")
    show(io, gp.k, 2)
    if (gp.nobsv == 0)
        println("  No observation data")
    else
        println(io, "  Input observations = ")
        show(io, gp.X)
        print(io,"\n  Output observations = ")
        show(io, gp.y)
        print(io,"\n  Variance of observation noise = $(exp(2*gp.logNoise))")
        print(io,"\n  Marginal Log-Likelihood = ")
        show(io, round(gp.target,3))
    end
end
