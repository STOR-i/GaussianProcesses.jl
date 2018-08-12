# Main GaussianProcess type

mutable struct GPMC{T<:Real} <: GPBase
    "Mean object"
    m:: Mean
    "Kernel object"
    k::Kernel
    "Likelihood"
    lik::Likelihood

    # Observation data
    "Number of observations"
    nobsv::Int
    "Input observations"
    X::MatF64
    "Output observations"
    y::Vector{T}
    "Latent (whitened) variables - N(0,1)"
    v::VecF64
    "Auxiliary observation data (to speed up calculations)"
    data::KernelData
    "Dimension of inputs"
    dim::Int

    # Auxiliary data
    μ::Vector{Float64}
    "`(k + exp(2*obsNoise))`"
    cK::AbstractPDMat
    "Log-likelihood"
    ll::Float64
    "Gradient of log-likelihood"
    dll::VecF64
    "Log-target (marginal log-likelihood + log priors)"
    target::Float64
    "Gradient of log-target (gradient of marginal log-likelihood + gradient of log priors)"
    dtarget::VecF64

    """
        GPMC(X, y, m, k, lik)
        GPMC(; m=MeanZero(), k=SE(0.0, 0.0), lik=Likelihood()) # observation-free constructor

    Fit a Gaussian process to a set of training points. The Gaussian process with
    non-Gaussian observations is defined in terms of its user-defined likelihood function,
    mean and covaiance (kernel) functions.

    The non-Gaussian likelihood is handled by a Monte Carlo method. The latent function
    values are represented by centered (whitened) variables ``f(x) = m(x) + Lv`` where
    ``v ∼ N(0, I)`` and ``LLᵀ = K_θ``.

    # Arguments:
    - `X::Matrix{Float64}`: Input observations
    - `y::Vector{Float64}`: Output observations
    - `m::Mean`: Mean function
    - `k::kernel`: Covariance function
    - `lik::likelihood`: Likelihood function
    """
    function GPMC{T}(X::MatF64, y::Vector{T}, m::Mean, k::Kernel,
                     lik::Likelihood) where T<:Real
        dim, nobsv = size(X)
        v = zeros(nobsv)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new{T}(m, k, lik, nobsv, X, y, v, KernelData(k, X), dim)
        initialise_target!(gp)
        return gp
    end
end

GPMC(X::MatF64, y::Vector{T}, meanf::Mean, kernel::Kernel, lik::Likelihood) where T<:Real =
    GPMC{T}(X, y, meanf, kernel, lik)

# # Convenience constructor
GP(X::MatF64, y::Vector{T}, m::Mean, k::Kernel, lik::Likelihood) where T<:Real =
    GPMC{T}(X, y, m, k, lik)

# Creates GP object for 1D case
GPMC(x::VecF64, y::Vector{T}, meanf::Mean, kernel::Kernel, lik::Likelihood) where T<:Real =
    GPMC{T}(x', y, meanf, kernel, lik)


GP(x::VecF64, y::Vector{T}, m::Mean, k::Kernel, lik::Likelihood) where T<:Real =
    GPMC{T}(x', y, m, k, lik)

"""
    initialise_ll!(gp::GPMC)

Initialise the log-likelihood of Gaussian process `gp`.
"""
function initialise_ll!(gp::GPMC)
    # log p(Y|v,θ)
    gp.μ = mean(gp.m,gp.X)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + 1e-6*I)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
end

"""
    update_cK!(gp::GPMC)

Update the covariance matrix and its Cholesky decomposition of Gaussian process `gp`.
"""
function update_cK!(gp::GPMC)
    old_cK = gp.cK
    Σbuffer = old_cK.mat
    cov!(Σbuffer, gp.k, gp.X, gp.data)
    for i in 1:gp.nobsv
        Σbuffer[i,i] += 1e-6 # no logNoise for GPMC
    end
    chol_buffer = old_cK.chol.factors
    copyto!(chol_buffer, Σbuffer)
    chol = cholesky!(Symmetric(chol_buffer))
    gp.cK = PDMats.PDMat(Σbuffer, chol)
end

# modification of initialise_ll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_ll!(gp::GPMC; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    if kern
        # only need to update the covariance matrix
        # if the covariance parameters have changed
        update_cK!(gp)
    end
    gp.μ = mean(gp.m,gp.X)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
end

"""
     update_dll!(gp::GPMC, ...)

Update the gradient of the log-likelihood of Gaussian process `gp`.
"""
function update_dll!(gp::GPMC, Kgrad::MatF64, L_bar::MatF64;
    process::Bool=true, # include gradient components for the process itself
    lik::Bool=true,  # include gradient components for the likelihood parameters
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    size(Kgrad) == (gp.nobsv, gp.nobsv) || throw(ArgumentError,
    @sprintf("Buffer for Kgrad should be a %dx%d matrix, not %dx%d",
             gp.nobsv, gp.nobsv,
             size(Kgrad,1), size(Kgrad,2)))
    size(L_bar) == (gp.nobsv, gp.nobsv) || throw(ArgumentError,
    @sprintf("Buffer for L_bar should be a %dx%d matrix, not %dx%d",
             gp.nobsv, gp.nobsv,
             size(L_bar,1), size(L_bar,2)))

    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    gp.dll = Array{Float64}(undef, process * gp.nobsv + lik * n_lik_params +
                            domean * n_mean_params + kern * n_kern_params)

    Lv = unwhiten(gp.cK, gp.v)
    dl_df = dlog_dens_df(gp.lik, Lv + gp.μ, gp.y)
    i = 1
    if process
        mul!(view(gp.dll, i:i+gp.nobsv-1), gp.cK.chol.U, dl_df)
        i += gp.nobsv
    end
    if lik && n_lik_params > 0
        Lgrads = dlog_dens_dθ(gp.lik, Lv + gp.μ, gp.y)
        for j in 1:n_lik_params
            gp.dll[i] = sum(Lgrads[:,j])
            i += 1
        end
    end
    if domean && n_mean_params > 0
        Mgrads = grad_stack(gp.m, gp.X)
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
            grad_slice!(Kgrad, gp.k, gp.X, gp.data, iparam)
            gp.dll[i] = dot(Kgrad, L_bar)
            i+=1
        end
    end
end

function update_ll_and_dll!(gp::GPMC, Kgrad::MatF64, L_bar::MatF64; kwargs...)
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
    gp.target = gp.ll - 0.5 * (sum(abs2, gp.v) + log(2 * pi) * length(gp.v)) +
        prior_logpdf(gp.lik) + prior_logpdf(gp.m) .+ prior_logpdf(gp.k)
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
    gp.target = gp.ll - 0.5 * (sum(abs2, gp.v) + log(2 * pi) * length(gp.v)) +
        prior_logpdf(gp.lik) + prior_logpdf(gp.m) + prior_logpdf(gp.k)
end

function update_dtarget!(gp::GPMC, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_dll!(gp, Kgrad, L_bar; kwargs...)
    gp.dtarget = gp.dll + prior_gradlogpdf(gp; kwargs...)
end

"""
    update_target_and_dtarget!(gp::GPMC, ...)

Update the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Gaussian process `gp` and its derivative.
"""
function update_target_and_dtarget!(gp::GPMC, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_target!(gp; kwargs...)
    update_dtarget!(gp, Kgrad, L_bar; kwargs...)
end

function update_target_and_dtarget!(gp::GPMC; kwargs...)
    Kgrad = Array{Float64}(undef, gp.nobsv, gp.nobsv)
    L_bar = Array{Float64}(undef, gp.nobsv, gp.nobsv)
    update_target_and_dtarget!(gp, Kgrad, L_bar; kwargs...)
end


"""
    predict_y(gp::GPMC, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])

Return the predictive mean and variance of Gaussian Process `gp` at specfic points which
are given as columns of matrix `x`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predict_y(gp::GPMC, x::MatF64; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    return predict_obs(gp.lik, μ, σ2)
end

# 1D Case for prediction
predict_y(gp::GPMC, x::VecF64; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)


## compute predictions
function _predict(gp::GPMC, X::MatF64)
    n = size(X, 2)
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    fmu =  mean(gp.m,X) + Lck'gp.v     # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Add jitter to get stable covariance
    fSigma = tolerant_PDMat(Sigma_raw)
    return fmu, fSigma
end


# Sample from functions from the GP
function Random.rand!(gp::GPMC, X::MatF64, A::DenseMatrix)
    nobsv = size(X,2)
    n_sample = size(A,2)

    if gp.nobsv == 0
        # Prior mean and covariance
        μ = mean(gp.m, X);
        Σraw = cov(gp.k, X);
        # Add jitter to get stable covariance
        Σ = tolerant_PDMat(Σraw)
    else
        # Posterior mean and covariance
        μ, Σ = predict_f(gp, X; full_cov=true)
    end

    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobsv, n_sample)))
end

#Samples random function values f
function Random.rand(gp::GPMC, X::MatF64, n::Int)
    nobsv = size(X, 2)
    A = Array{Float64}(undef, nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GP
Random.rand(gp::GPMC, x::VecF64, n::Int) = rand(gp, x', n)

# Generate only one sample from the GP and returns a vector
Random.rand(gp::GPMC, X::MatF64) = vec(rand(gp,X,1))
Random.rand(gp::GPMC, x::VecF64) = vec(rand(gp,x',1))


function get_params(gp::GPMC; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    append!(params, gp.v)
    if lik  && num_params(gp.lik)>0
        append!(params, get_params(gp.lik))
    end
    if domean && num_params(gp.m)>0
        append!(params, get_params(gp.m))
    end
    if kern
        append!(params, get_params(gp.k))
    end
    return params
end

function num_params(gp::GPMC; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n = length(gp.v)
    lik && (n += num_params(gp.lik))
    domean && (n += num_params(gp.m))
    kern && (n += num_params(gp.k))
    n
end

function set_params!(gp::GPMC, hyp::VecF64; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    i = 1
    if process
        gp.v = hyp[1:gp.nobsv]
        i += gp.nobsv
    end
    if lik  && n_lik_params>0
        set_params!(gp.lik, hyp[i:i+n_lik_params-1]);
        i += n_lik_params
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
        append!(grad, prior_gradlogpdf(gp.m))
    end
    if kern
        append!(grad, prior_gradlogpdf(gp.k))
    end
    return grad
end


function Base.show(io::IO, gp::GPMC)
    println(io, "GP Monte Carlo object:")
    println(io, "  Dim = $(gp.dim)")
    println(io, "  Number of observations = $(gp.nobsv)")
    println(io, "  Mean function:")
    show(io, gp.m, 2)
    println(io, "  Kernel:")
    show(io, gp.k, 2)
    println(io, "  Likelihood:")
    show(io, gp.lik, 2)
    if (gp.nobsv == 0)
        println("  No observation data")
    else
        println(io, "  Input observations = ")
        show(io, gp.X)
        print(io,"\n  Output observations = ")
        show(io, gp.y)
        print(io,"\n  Log-posterior = ")
        show(io, round(gp.target; digits=3))
    end
end

