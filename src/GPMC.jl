# Main GaussianProcess type

mutable struct GPMC{X<:AbstractMatrix,Y<:AbstractVector{<:Real},M<:Mean,K<:Kernel,L<:Likelihood,
                    D<:KernelData} <: GPBase
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
    "Likelihood"
    lik::L

    # Auxiliary data
    "Dimension of inputs"
    dim::Int
    "Number of observations"
    nobs::Int
    "Auxiliary observation data (to speed up calculations)"
    data::D
    "Latent (whitened) variables - N(0,1)"
    v::Vector{Float64}
    "Mean values"
    μ::Vector{Float64}
    "`(k + exp(2*obsNoise))`"
    cK::PDMat{Float64,Matrix{Float64}}
    "Log-likelihood"
    ll::Float64
    "Gradient of log-likelihood"
    dll::Vector{Float64}
    "Log-target (marginal log-likelihood + log priors)"
    target::Float64
    "Gradient of log-target (gradient of marginal log-likelihood + gradient of log priors)"
    dtarget::Vector{Float64}

    function GPMC{X,Y,M,K,L}(x::X, y::Y, mean::M, kernel::K, lik::L) where {X,Y,M,K,L}
        dim, nobs = size(x)
        length(y) == nobs || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        data = KernelData(kernel, x, x)
        gp = new{X,Y,M,K,L,typeof(data)}(x, y, mean, kernel, lik, dim, nobs, data,
                                         zeros(nobs))
        initialise_target!(gp)
    end
end

"""
    GPMC(x, y, mean, kernel, lik)

Fit a Gaussian process to a set of training points. The Gaussian process with
non-Gaussian observations is defined in terms of its user-defined likelihood function,
mean and covaiance (kernel) functions.

The non-Gaussian likelihood is handled by a Monte Carlo method. The latent function
values are represented by centered (whitened) variables ``f(x) = m(x) + Lv`` where
``v ∼ N(0, I)`` and ``LLᵀ = K_θ``.

# Arguments:
- `x::AbstractVecOrMat{Float64}`: Input observations
- `y::AbstractVector{<:Real}`: Output observations
- `mean::Mean`: Mean function
- `kernel::Kernel`: Covariance function
- `lik::Likelihood`: Likelihood function
"""
GPMC(x::AbstractMatrix, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood) =
    GPMC{typeof(x),typeof(y),typeof(mean),typeof(kernel),typeof(lik)}(x, y, mean, kernel,
                                                                      lik)

GPMC(x::AbstractVector, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel, lik::Likelihood) =
    GPMC(x', y, mean, kernel, lik)

"""
    GP(x, y, mean::Mean, kernel::Kernel, lik::Likelihood)

Fit a Gaussian process that is defined by its `mean`, its `kernel`, and its likelihood
function `lik` to a set of training points `x` and `y`.

See also: [`GPMC`](@ref)
"""
GP(x::AbstractVecOrMat{Float64}, y::AbstractVector{<:Real}, mean::Mean, kernel::Kernel,
   lik::Likelihood) = GPMC(x, y, mean, kernel, lik)

"""
    initialise_ll!(gp::GPMC)

Initialise the log-likelihood of Gaussian process `gp`.
"""
function initialise_ll!(gp::GPMC)
    # log p(Y|v,θ)
    gp.μ = mean(gp.mean,gp.x)
    Σ = cov(gp.kernel, gp.x, gp.data)
    gp.cK = PDMat(Σ + 1e-6*I)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
    gp
end

"""
    update_cK!(gp::GPMC)

Update the covariance matrix and its Cholesky decomposition of Gaussian process `gp`.
"""
function update_cK!(gp::GPMC)
    old_cK = gp.cK
    Σbuffer = old_cK.mat
    cov!(Σbuffer, gp.kernel, gp.x, gp.data)
    for i in 1:gp.nobs
        Σbuffer[i,i] += 1e-6 # no logNoise for GPMC
    end
    chol_buffer = old_cK.chol.factors
    copyto!(chol_buffer, Σbuffer)
    chol = cholesky!(Symmetric(chol_buffer))
    gp.cK = PDMat(Σbuffer, chol)
    gp
end

# modification of initialise_ll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_ll!(gp::GPMC; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    if kern
        # only need to update the covariance matrix
        # if the covariance parameters have changed
        update_cK!(gp)
    end
    gp.μ = mean(gp.mean,gp.x)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
    gp
end

"""
     update_dll!(gp::GPMC, ...)

Update the gradient of the log-likelihood of Gaussian process `gp`.
"""
function update_dll!(gp::GPMC, Kgrad::AbstractMatrix, L_bar::AbstractMatrix;
    process::Bool=true, # include gradient components for the process itself
    lik::Bool=true,  # include gradient components for the likelihood parameters
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    size(Kgrad) == (gp.nobs, gp.nobs) || throw(ArgumentError,
    @sprintf("Buffer for Kgrad should be a %dx%d matrix, not %dx%d",
             gp.nobs, gp.nobs,
             size(Kgrad,1), size(Kgrad,2)))
    size(L_bar) == (gp.nobs, gp.nobs) || throw(ArgumentError,
    @sprintf("Buffer for L_bar should be a %dx%d matrix, not %dx%d",
             gp.nobs, gp.nobs,
             size(L_bar,1), size(L_bar,2)))

    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)

    gp.dll = Array{Float64}(undef, process * gp.nobs + lik * n_lik_params +
                            domean * n_mean_params + kern * n_kern_params)

    Lv = unwhiten(gp.cK, gp.v)
    dl_df = dlog_dens_df(gp.lik, Lv + gp.μ, gp.y)
    i = 1
    if process
        mul!(view(gp.dll, i:i+gp.nobs-1), gp.cK.chol.U, dl_df)
        i += gp.nobs
    end
    if lik && n_lik_params > 0
        Lgrads = dlog_dens_dθ(gp.lik, Lv + gp.μ, gp.y)
        for j in 1:n_lik_params
            gp.dll[i] = sum(Lgrads[:,j])
            i += 1
        end
    end
    if domean && n_mean_params > 0
        Mgrads = grad_stack(gp.mean, gp.x)
        for j in 1:n_mean_params
            gp.dll[i] = dot(dl_df,Mgrads[:,j])
            i += 1
        end
    end
    if kern
        fill!(L_bar, 0)
        BLAS.ger!(1.0, dl_df, gp.v, L_bar)
        tril!(L_bar)
        # ToDo:
        # the following two steps allocates memory
        # and are fickle, reaching into the internal
        # implementation of the cholesky decomposition
        L = gp.cK.chol.L.data
        tril!(L)
        #
        chol_unblocked_rev!(L, L_bar)
        for iparam in 1:n_kern_params
            grad_slice!(Kgrad, gp.kernel, gp.x, gp.data, iparam)
            gp.dll[i] = dot(Kgrad, L_bar)
            i+=1
        end
    end

    gp
end

function update_ll_and_dll!(gp::GPMC, Kgrad::AbstractMatrix, L_bar::AbstractMatrix; kwargs...)
    update_ll!(gp; kwargs...)
    update_dll!(gp, Kgrad, L_bar; kwargs...)
end


"""
    initialise_target!(gp::GPMC)

Initialise the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Gaussian process `gp`.
"""
function initialise_target!(gp::GPMC)
    initialise_ll!(gp)
    gp.target = gp.ll - (sum(abs2, gp.v) + log2π * gp.nobs) / 2 +
        prior_logpdf(gp.lik) + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel)
    gp
end

"""
    update_target!(gp::GPMC, ...)

Update the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Gaussian process `gp`.
"""
function update_target!(gp::GPMC; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    update_ll!(gp; process=process, lik=lik, domean=domean, kern=kern)
    gp.target = gp.ll - (sum(abs2, gp.v) + log2π * gp.nobs) / 2 +
        prior_logpdf(gp.lik) + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel)
    gp
end

function update_dtarget!(gp::GPMC, Kgrad::AbstractMatrix, L_bar::AbstractMatrix; kwargs...)
    update_dll!(gp, Kgrad, L_bar; kwargs...)
    gp.dtarget = gp.dll + prior_gradlogpdf(gp; kwargs...)
    gp
end

"""
    update_target_and_dtarget!(gp::GPMC, ...)

Update the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Gaussian process `gp` and its derivative.
"""
function update_target_and_dtarget!(gp::GPMC, Kgrad::AbstractMatrix, L_bar::AbstractMatrix; kwargs...)
    update_target!(gp; kwargs...)
    update_dtarget!(gp, Kgrad, L_bar; kwargs...)
end

function update_target_and_dtarget!(gp::GPMC; kwargs...)
    Kgrad = Array{Float64}(undef, gp.nobs, gp.nobs)
    L_bar = Array{Float64}(undef, gp.nobs, gp.nobs)
    update_target_and_dtarget!(gp, Kgrad, L_bar; kwargs...)
end


"""
    predict_y(gp::GPMC, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])

Return the predictive mean and variance of Gaussian Process `gp` at specfic points which
are given as columns of matrix `x`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_y(gp::GPMC, x::AbstractMatrix; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    return predict_obs(gp.lik, μ, σ2)
end

# 1D Case for prediction
predict_y(gp::GPMC, x::AbstractVector; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)


## compute predictions
function _predict(gp::GPMC, x::AbstractMatrix)
    cK = cov(gp.kernel, gp.x, x)
    Lck = whiten(gp.cK, cK)
    fmu =  mean(gp.mean,x) + Lck'gp.v     # Predictive mean
    Sigma_raw = cov(gp.kernel, x) - Lck'Lck # Predictive covariance
    # Add jitter to get stable covariance
    Σ, chol = make_posdef!(Sigma_raw)
    return fmu, Σ
end


# Sample from functions from the GP
function Random.rand!(gp::GPMC, x::AbstractMatrix, A::DenseMatrix)
    nobs = size(x,2)
    n_sample = size(A,2)

    if gp.nobs == 0
        # Prior mean and covariance
        μ = mean(gp.mean, x);
        Σraw = cov(gp.kernel, x);
        # Add jitter to get stable covariance
        Σ, chol = make_posdef!(Σraw)
    else
        # Posterior mean and covariance
        μ, Σ = predict_f(gp, x; full_cov=true)
    end
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobs, n_sample)))
end

Random.rand(gp::GPMC, x::AbstractMatrix, n::Int) = rand!(gp, x, Array{Float64}(undef, size(x, 2), n))

# Sample from 1D GP
Random.rand(gp::GPMC, x::AbstractVector, n::Int) = rand(gp, x', n)

# Generate only one sample from the GP and returns a vector
Random.rand(gp::GPMC, x::AbstractMatrix) = vec(rand(gp,x,1))
Random.rand(gp::GPMC, x::AbstractVector) = vec(rand(gp,x',1))

appendlikbounds!(lb, ub, gp::GPMC, bounds) = appendbounds!(lb, ub, num_params(gp.lik), bounds)

function get_params(gp::GPMC; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    append!(params, gp.v)
    if lik  && num_params(gp.lik)>0
        append!(params, get_params(gp.lik))
    end
    if domean && num_params(gp.mean)>0
        append!(params, get_params(gp.mean))
    end
    if kern
        append!(params, get_params(gp.kernel))
    end
    return params
end

function num_params(gp::GPMC; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n = length(gp.v)
    lik && (n += num_params(gp.lik))
    domean && (n += num_params(gp.mean))
    kern && (n += num_params(gp.kernel))
    n
end

function set_params!(gp::GPMC, hyp::AbstractVector; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)

    i = 1
    if process
        gp.v = hyp[1:gp.nobs]
        i += gp.nobs
    end
    if lik  && n_lik_params>0
        set_params!(gp.lik, hyp[i:i+n_lik_params-1]);
        i += n_lik_params
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

function prior_gradlogpdf(gp::GPMC; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    if process
        grad = -gp.v
    else
        grad = Float64[]
    end
    if lik
        append!(grad, prior_gradlogpdf(gp.lik))
    end
    if domean
        append!(grad, prior_gradlogpdf(gp.mean))
    end
    if kern
        append!(grad, prior_gradlogpdf(gp.kernel))
    end
    return grad
end


function Base.show(io::IO, gp::GPMC)
    println(io, "GP Monte Carlo object:")
    println(io, "  Dim = ", gp.dim)
    println(io, "  Number of observations = ", gp.nobs)
    println(io, "  Mean function:")
    show(io, gp.mean, 2)
    println(io, "\n  Kernel:")
    show(io, gp.kernel, 2)
    println(io, "\n  Likelihood:")
    show(io, gp.lik, 2)
    if gp.nobs == 0
        println("\n  No observation data")
    else
        println(io, "\n  Input observations = ")
        show(io, gp.x)
        print(io,"\n  Output observations = ")
        show(io, gp.y)
        print(io,"\n  Log-posterior = ")
        show(io, round(gp.target; digits = 3))
    end
end
