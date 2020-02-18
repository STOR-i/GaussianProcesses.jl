mutable struct SSGP{X<:AbstractMatrix,Y<:AbstractVector{<:Real}, F<:RFF, M<:Mean, K<:Kernel, L<:Likelihood,
                    CS<:CovarianceStrategy, D<:KernelData} <: GPBase
    # Observation data
    "Input observations"
    x::X
    "Output observations"
    y::Y
    fourier::F
    # Model
    "Mean object"
    mean::M
    "Kernel object"
    kernel::K
    "Likelihood"
    lik::L
    "Strategy for computing or approximating covariance matrices"
    covstrat::CS
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
    "Cholesky of the scaled ϕ⋅ϕ''"
    R::Cholesky

    function SSGP{X,Y,F,M,K,L,CS,D}(x::X, y::Y, fourier::F, mean::M, kernel::K, lik::L, covstrat::CS, data::D) where {X,Y,F,M,K,L,CS,D}
        dim, nobs = size(x)
        length(y) == nobs || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new{X,Y,F,M,K,L,CS,D}(x, y, fourier, mean, kernel, lik, covstrat, dim, nobs, data, zeros(nobs))
        initialise_target!(gp)
    end
end

function SSGP(x::AbstractMatrix, y::AbstractVector{<:Real}, F::RFF, mean::Mean, kernel::Kernel, lik::Likelihood, covstrat::CovarianceStrategy)
    data = KernelData(kernel, x, x, covstrat)
    return SSGP{typeof(x),typeof(y), typeof(F), typeof(mean), typeof(kernel), typeof(lik), typeof(covstrat), typeof(data)}(
                x, y, F, mean, kernel, lik, covstrat, data)
end

"""
    SSGP(x, y, mean, kernel, lik)

Fit a Gaussian process to a set of training points. The Gaussian process with
non-Gaussian observations is defined in terms of its user-defined likelihood function,
mean and covaiance (kernel) functions.

The non-Gaussian likelihood is handled by an approximate method (e.g. Monte Carlo). The latent function
values are represented by centered (whitened) variables ``f(x) = m(x) + Lv`` where
``v ∼ N(0, I)`` and ``LLᵀ = K_θ``.

# Arguments:
- `x::AbstractVecOrMat{Float64}`: Input observations
- `y::AbstractVector{<:Real}`: Output observations
- `mean::Mean`: Mean function
- `kernel::Kernel`: Covariance function
- `lik::Likelihood`: Likelihood function
"""
function SSGP(x::AbstractMatrix, y::AbstractVector{<:Real}, F::RFF, mean::Mean, kernel::Kernel, lik::Likelihood)
    covstrat = FullCovariance()
    return SSGP(x, y, F, mean, kernel, lik, covstrat)
end

SSGP(x::AbstractVector, y::AbstractVector{<:Real}, F::RFF, mean::Mean, kernel::Kernel, lik::Likelihood) =
    SSGP(x', y, F, mean, kernel, lik)

"""
    initialise_ll!(gp::SSGP)

Initialise the log-likelihood of Gaussian process `gp`.
"""
function initialise_ll!(gp::SSGP)
    # log p(Y|v,θ)
    gp.μ = mean(gp.mean,gp.x)
    Σ = cov(gp.kernel, gp.x, gp.x, gp.data)
    gp.cK = PDMat(Σ + 1e-6*I)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
    gp
end

"""
    update_cK!(gp::SSGP)

Update the covariance matrix and its Cholesky decomposition of Gaussian process `gp`.
"""
function update_cK!(gp::SSGP)
    old_cK = gp.cK
    Σbuffer = old_cK.mat
    cov!(Σbuffer, gp.kernel, gp.x, gp.x, gp.data)
    for i in 1:gp.nobs
        Σbuffer[i,i] += 1e-6 # no logNoise for SSGP
    end
    chol_buffer = old_cK.chol.factors
    copyto!(chol_buffer, Σbuffer)
    chol = cholesky!(Symmetric(chol_buffer))
    gp.cK = PDMat(Σbuffer, chol)
    gp
end

# modification of initialise_ll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_ll!(gp::SSGP; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
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

init_precompute(gp::SSGP) = FullCovMCMCPrecompute(gp.nobs)

"""
     update_dll!(gp::SSGP, ...)

Update the gradient of the log-likelihood of Gaussian process `gp`.
"""
function update_dll!(gp::SSGP, precomp::AbstractGradientPrecompute;
    process::Bool=true, # include gradient components for the process itself
    lik::Bool=true,  # include gradient components for the likelihood parameters
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )

    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.mean)
    n_kern_params = num_params(gp.kernel)

    gp.dll = Array{Float64}(undef, process * gp.nobs + lik * n_lik_params +
                            domean * n_mean_params + kern * n_kern_params)
    precompute!(precomp, gp)

    i = 1
    if process
        mul!(view(gp.dll, i:i+gp.nobs-1), gp.cK.chol.U, precomp.dl_df)
        i += gp.nobs
    end
    if lik && n_lik_params > 0
        Lgrads = dlog_dens_dθ(gp.lik, precomp.f, gp.y)
        for j in 1:n_lik_params
            gp.dll[i] = sum(Lgrads[:,j])
            i += 1
        end
    end
    # if domean && n_mean_params > 0
        # Mgrads = grad_stack(gp.mean, gp.x)
        # for j in 1:n_mean_params
            # gp.dll[i] = dot(precomp.dl_df,Mgrads[:,j])
            # i += 1
        # end
    # end
    if domean && n_mean_params>0
        dll_m = @view(gp.dll[i:i+n_mean_params-1])
        dll_mean!(dll_m, gp, precomp)
        i += n_mean_params
    end
    if kern
        dll_k = @view(gp.dll[i:end])
        dll_kern!(dll_k, gp, precomp, gp.covstrat)
    end

    gp
end

function update_ll_and_dll!(gp::SSGP, precomp::AbstractGradientPrecompute; kwargs...)
    update_ll!(gp; kwargs...)
    update_dll!(gp, precomp; kwargs...)
end


"""
    initialise_target!(gp::SSGP)

Initialise the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Gaussian process `gp`.
"""
function initialise_target!(gp::SSGP)
    initialise_ll!(gp)
    gp.target = gp.ll - (sum(abs2, gp.v) + log2π * gp.nobs) / 2 +
        prior_logpdf(gp.lik) + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel)
    gp
end

"""
    update_target!(gp::SSGP, ...)

Update the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Gaussian process `gp`.
"""
function update_target!(gp::SSGP; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    update_ll!(gp; process=process, lik=lik, domean=domean, kern=kern)
    gp.target = gp.ll - (sum(abs2, gp.v) + log2π * gp.nobs) / 2 +
        prior_logpdf(gp.lik) + prior_logpdf(gp.mean) + prior_logpdf(gp.kernel)
    gp
end

function update_dtarget!(gp::SSGP, precomp::AbstractGradientPrecompute; kwargs...)
    update_dll!(gp, precomp; kwargs...)
    gp.dtarget = gp.dll + prior_gradlogpdf(gp; kwargs...)
    gp
end

"""
    update_target_and_dtarget!(gp::SSGP, ...)

Update the log-posterior
```math
\\log p(θ, v | y) ∝ \\log p(y | v, θ) + \\log p(v) + \\log p(θ)
```
of a Gaussian process `gp` and its derivative.
"""
function update_target_and_dtarget!(gp::SSGP, precomp::AbstractGradientPrecompute; kwargs...)
    update_target!(gp; kwargs...)
    update_dtarget!(gp, precomp; kwargs...)
end

function update_target_and_dtarget!(gp::SSGP; kwargs...)
    precomp = init_precompute(gp)
    update_target_and_dtarget!(gp, precomp; kwargs...)
end


predict_full(gp::SSGP, xpred::AbstractMatrix) = predictMVN(xpred, gp.x, gp.y, gp.kernel, gp.mean, gp.cK\unwhiten(gp.cK,gp.v), gp.covstrat, gp.cK)
"""
    predict_y(gp::SSGP, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])

Return the predictive mean and variance of Gaussian Process `gp` at specfic points which
are given as columns of matrix `x`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""

function predict_y(gp::SSGP, x::AbstractMatrix)
    ϕ = build_design_mat(gp.fourier, gp.x) # N x M matric
    norm = ((gp.fourier.σ^2)/gp.fourier.M) # Constant
    # TODO: Add a conditional in to prevent repetitive computation of R
    R = cholesky(norm * (ϕ * ϕ') + (I(gp.nobs) * 1e-10))
    gp.R = R
    Ry = reshape(R\gp.y, (size(R, 1), 1))
    α = reshape(R.L\gp.y, size(R, 1), 1)
    # println("X: ", size(x))
    # println("α: ", size(α))
    # println("norm: ", norm)
    # println("R: ", size(R))
    # println("Ry: ", size(Ry))
    # println("ϕ: ", size(ϕ))
    ϕs = build_design_mat(gp.fourier, x)
    # println("ϕs: ", size(ϕs))
    # println("ϕ * ϕs': ", size(ϕ*ϕs'))
    μ = norm * ((ϕ* ϕs')' * α)
    # println("μ: ", size(μ))
    β = R\(ϕ *ϕs')
    # println("β: ", size(β))
    # println("ϕs: ", size(ϕs))
    # println("ϕs * ϕs': ", size(ϕs * ϕs'))
    # println("β' * β: ", size(β' * β))
    # println("norm * diag: ",  size((norm * diag(ϕs * ϕs'))))
    Σ = sqrt.(norm * diag((ϕs * ϕs') - norm*(β' * β))) # want 30 x 30
    return μ, Σ
end

appendlikbounds!(lb, ub, gp::SSGP, bounds) = appendbounds!(lb, ub, num_params(gp.lik), bounds)

#—————————————————————————————————————————————————————–
# Function for sampling from the prior of the GP object hyperparameters.

function sample_params(gp::SSGP; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    samples = Float64[]
    if lik && num_params(gp.lik)>0
        like_priors = get_priors(gp.lik)
        @assert !isempty(like_priors) "prior distributions of likelihood hyperparameters should be set"
        like_sample = rand(Product(like_priors))
        append!(samples, like_sample)
    end
    if domean && num_params(gp.mean)>0
        mean_priors = get_priors(gp.mean)
        @assert !isempty(mean_priors) "prior distributions of mean hyperparameters should be set"
        mean_sample = rand(Product(mean_priors))
        append!(samples, mean_sample)
    end
    if kern && num_params(gp.kernel)>0
        kernel_priors = get_priors(gp.kernel)
        @assert !isempty(kernel_priors) "prior distributions of kernel hyperparameters should be set"
        kernel_sample = rand(Product(kernel_priors))
        append!(samples, kernel_sample)
    end
    return samples
end

function get_params(gp::SSGP; lik::Bool=true, domean::Bool=true, kern::Bool=true)
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

function num_params(gp::SSGP; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n = length(gp.v)
    lik && (n += num_params(gp.lik))
    domean && (n += num_params(gp.mean))
    kern && (n += num_params(gp.kernel))
    n
end

function set_params!(gp::SSGP, hyp::AbstractVector; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
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

function prior_gradlogpdf(gp::SSGP; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
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


function Base.show(io::IO, gp::SSGP)
    println(io, "GP Approximate object:")
    println(io, "  Dim = ", gp.dim)
    println(io, "  Number of observations = ", gp.nobs)
    println(io, "  Number of Fourier features = ", gp.fourier.M)
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
