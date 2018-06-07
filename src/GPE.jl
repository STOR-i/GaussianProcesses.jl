import Base.show
# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.

# Constructors:
    GPE(X, y, m, k, logNoise)
    GPE(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-2.0) # observation-free constructor

# Arguments:
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `logNoise::Float64` : Log of the standard deviation for the observation noise. The default is -2.0, which is equivalent to assuming no observation noise.

# Returns:
* `gp::GPE`            : Gaussian process object, fitted to the training data if provided
""" ->
type GPE <: GPBase
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    logNoise::Float64       # log standard deviation of observation noise
    
    # Observation data
    nobsv::Int              # Number of observations
    X::MatF64               # Input observations
    y::Vector{Float64}      # Output observations
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    cK::AbstractPDMat       # (k + exp(2*obsNoise))
    alpha::Vector{Float64}  # (k + exp(2*obsNoise))⁻¹y
    mll::Float64            # Marginal log-likelihood
    target::Float64            # Log target (Marginal log-likelihood + log priors)
    dmll::Vector{Float64}      # Gradient of marginal log-likelihood
    dtarget::Vector{Float64}   # Gradient log-target (gradient of marginal log-likelihood + gradient of log priors)
    
    function GPE(X::MatF64, y::Vector{Float64}, m::Mean, k::Kernel, logNoise::Float64=-2.0)
        dim, nobsv = size(X)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(m, k, logNoise, nobsv, X, y, KernelData(k, X), dim)
        initialise_target!(gp)
        return gp
    end
    
    GPE(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-2.0) =  new(m, k, logNoise, 0)
    GPE(m::Mean, k::Kernel, logNoise::Float64, nobsv::Int,
        X::MatF64, y::Vector{Float64},
        data::KernelData, dim::Int,
        cK::AbstractPDMat, alpha::Vector{Float64},
        mll::Float64, target::Float64,
        dmll::Vector{Float64}, dtarget::Vector{Float64}) = new(
                m, k, logNoise, nobsv,
                X, y, data, dim, cK, alpha,
                mll, target, dmll, dtarget)
end

GP(X::Matrix{Float64}, y::Vector{Float64}, m::Mean, k::Kernel, logNoise::Float64=-2.0) = GPE(X, y, m, k, logNoise)

# Creates GPE object for 1D case
GPE(x::Vector{Float64}, y::Vector{Float64}, meanf::Mean, kernel::Kernel, logNoise::Float64=-2.0) = GPE(x', y, meanf, kernel, logNoise)

GP(x::Vector{Float64}, y::Vector{Float64}, m::Mean, k::Kernel, logNoise::Float64=-2.0) = GPE(x', y, m, k, logNoise)


@doc """
# Description
Fits an existing Gaussian process to a set of training points.

# Arguments:
* `gp::GPE`: Exiting Gaussian process object
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations

# Returns:
* `gp::GPE`            : A Gaussian process fitted to the training data
""" ->
function fit!(gp::GPE, X::MatF64, y::Vector{Float64})
    length(y) == size(X,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.X = X
    gp.y = y
    gp.data = KernelData(gp.k, X)
    gp.dim, gp.nobsv = size(X)
    initialise_target!(gp)
    return gp
end

fit!(gp::GPE, x::Vector{Float64}, y::Vector{Float64}) = fit!(gp, x', y)

#———————————————————————————————————————————————————————————
#Fast memory allocation function

@doc """
get_ααinvcKI!(ααinvcKI::Matrix, cK::AbstractPDMat, α::Vector)

# Description
Computes α*α'-cK\eye(nobsv) in-place, avoiding any memory allocation

# Arguments:
* `ααinvcKI::Matrix` the matrix to be overwritten
* `cK::AbstractPDMat` the covariance matrix of the GPE (supplied by gp.cK)
* `α::Vector` the alpha vector of the GPE (defined as cK \ (Y-μ), and supplied by gp.alpha)
* nobsv
"""
function get_ααinvcKI!{M<:MatF64}(ααinvcKI::M, cK::AbstractPDMat, α::Vector)
    nobsv = length(α)
    size(ααinvcKI) == (nobsv, nobsv) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         nobsv, nobsv,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    ααinvcKI[:,:] = 0.0
    @inbounds for i in 1:nobsv
        ααinvcKI[i,i] = -1.0
    end
    A_ldiv_B!(cK.chol, ααinvcKI)
    LinAlg.BLAS.ger!(1.0, α, α, ααinvcKI)
end

#———————————————————————————————————————————————————————————————-
#Functions for calculating the log-target

""" GPE: Initialise the marginal log-likelihood """
function initialise_mll!(gp::GPE)
    μ = mean(gp.m,gp.X)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + (exp(2*gp.logNoise) + 1e-5)*I)
    gp.alpha = gp.cK \ (gp.y - μ)
    gp.mll = -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end    

"""
Update the covariance matrix and its Cholesky decomposition.
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
    copy!(chol_buffer, Σbuffer)
    chol = cholfact!(Symmetric(chol_buffer))
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

function dmll_kern!(dmll::VecF64, k::Kernel, X::MatF64, data::KernelData, ααinvcKI::MatF64)
    dim = size(X, 1)
    nparams = num_params(k)
    @assert nparams == length(dmll)
    nobsv = size(X, 2)
    dmll[:] = 0.0
    @inbounds for j in 1:nobsv
        # off-diagonal
        for i in 1:j-1
            for iparam in 1:nparams
                dKij = dKij_dθp(k,X,data,i,j,iparam,dim)
                dmll[iparam] += dKij * ααinvcKI[i, j]
            end
        end
        # diagonal
        for iparam in 1:nparams
            dKjj = dKij_dθp(k,X,data,j,j,iparam,dim)
            dmll[iparam] += (dKjj * ααinvcKI[j, j]) / 2.0
        end
    end
    return dmll
end
""" GPE: Update gradient of marginal log-likelihood """
function update_dmll!(gp::GPE,
    ααinvcKI::MatF64
    ; 
    noise::Bool=true, # include gradient component for the logNoise term
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    size(ααinvcKI) == (gp.nobsv, gp.nobsv) || throw(ArgumentError(
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         gp.nobsv, gp.nobsv,
                         size(ααinvcKI,1), size(ααinvcKI,2))))
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)
    gp.dmll = Array{Float64}( noise + domean*n_mean_params + kern*n_kern_params)

    get_ααinvcKI!(ααinvcKI, gp.cK, gp.alpha)

    i=1
    if noise
        gp.dmll[i] = exp(2.0*gp.logNoise)*trace(ααinvcKI)
        i+=1
    end

    if domean && n_mean_params>0
        Mgrads = grad_stack(gp.m, gp.X)
        for j in 1:n_mean_params
            gp.dmll[i] = dot(Mgrads[:,j],gp.alpha)
            i += 1
        end
    end
    if kern
        dmll_k = @view(gp.dmll[i:end])
        dmll_kern!(dmll_k, gp.k, gp.X, gp.data, ααinvcKI)
    end
end

""" GPE: Update gradient of marginal log-likelihood """
function update_mll_and_dmll!(gp::GPE,
        ααinvcKI::MatF64;
        kwargs...)
    update_mll!(gp; kwargs...)
    update_dmll!(gp, ααinvcKI; kwargs...)
end


""" GPE: Initialise the target, which is assumed to be the log-posterior, log p(θ|y) ∝ log p(y|θ) +  log p(θ) """
function initialise_target!(gp::GPE)
    initialise_mll!(gp)
        #HOW TO SET-UP A PRIOR FOR THE LOGNOISE?
    gp.target = gp.mll   + prior_logpdf(gp.m) + prior_logpdf(gp.k)
end

""" GPE: Update the target, which is assumed to be the log-posterior, log p(θ|y) ∝ log p(y|θ) + log p(θ) """ 
function update_target!(gp::GPE; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    update_mll!(gp; noise=noise, domean=domean, kern=kern)
    #HOW TO SET-UP A PRIOR FOR THE LOGNOISE?
    gp.target = gp.mll  + prior_logpdf(gp.m) + prior_logpdf(gp.k)
end

function update_dtarget!(gp::GPE, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_dmll!(gp, L_bar; kwargs...)
    gp.dtarget = gp.dmll + prior_gradlogpdf(gp; kwargs...)
end


""" GPE: A function to update the target (aka log-posterior) and its derivative """
function update_target_and_dtarget!(gp::GPE, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_target!(gp; kwargs...)
    update_dtarget!(gp, Kgrad, L_bar; kwargs...)
end

""" GPE: A function to update the target (aka log-posterior) and its derivative """
function update_target_and_dtarget!(gp::GPE; kwargs...)
    ααinvcKI = Array{Float64}(gp.nobsv, gp.nobsv)
    update_target!(gp; kwargs...)
    update_dmll!(gp, ααinvcKI; kwargs...)
    gp.dtarget = gp.dmll + prior_gradlogpdf(gp; kwargs...)
end


#——————————————————————#
# Predict observations #
#——————————————————————#

"""Calculate the mean and variance of predictive distribution p(y^*|x^*,D,θ) at test locations x^* """
function predict_y{M<:MatF64}(gp::GPE, x::M; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    if full_cov
        return μ, σ2 + ScalMat(gp.nobsv, exp(2*gp.logNoise))
    else
        return μ, σ2 + exp(2*gp.logNoise)
    end
end

# 1D Case for predictions
predict_y{V<:VecF64}(gp::GPE, x::V; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)

@deprecate predict predict_y

## compute predictions
function _predict{M<:MatF64}(gp::GPE, X::M)
    n = size(X, 2)
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    mu = mean(gp.m,X) + cK*gp.alpha        # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Add jitter to get stable covariance
    Sigma = tolerant_PDMat(Sigma_raw)
    return mu, Sigma
end

#———————————————————————————————————————————————————————————
# Sample from the GPE 
function rand!{M<:MatF64}(gp::GPE, X::M, A::DenseMatrix)
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
    
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobsv, n_sample)))
end

function rand{M<:MatF64}(gp::GPE, X::M, n::Int)
    nobsv=size(X,2)
    A = Array{Float64}( nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GPE
rand{V<:VecF64}(gp::GPE, x::V, n::Int) = rand(gp, x', n)

# Generate only one sample from the GPE and returns a vector
rand{M<:MatF64}(gp::GPE,X::M) = vec(rand(gp,X,1))
rand{V<:VecF64}(gp::GPE,x::V) = vec(rand(gp,x',1))

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

function set_params!(gp::GPE, hyp::Vector{Float64}; noise::Bool=true, domean::Bool=true, kern::Bool=true)
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
function push!(gp::GPE, X::MatF64, y::Vector{Float64})
    warn("push! method is currently inefficient as it refits all observations")
    if gp.nobsv == 0
        GaussianProcesses.fit!(gp, X, y)
    elseif size(X,1) != size(gp.X,1)
        error("New input observations must have dimensions consistent with existing observations")
    else
        GaussianProcesses.fit!(gp, cat(2, gp.X, X), cat(1, gp.y, y))
    end
end

push!(gp::GPE, x::Vector{Float64}, y::Vector{Float64}) = push!(gp, x', y)
push!(gp::GPE, x::Float64, y::Float64) = push!(gp, [x], [y])
push!(gp::GPE, x::Vector{Float64}, y::Float64) = push!(gp, reshape(x, length(x), 1), [y])



#—————————————————————————————————————————————————————————————
#Show function
function show(io::IO, gp::GPE)
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
