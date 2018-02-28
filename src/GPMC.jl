import Base.show
# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process, with non-Gaussian observations, is defined in terms of its likelihood function, mean and covaiance (kernel) functions, which are user defined. We use a Monte Carlo method to handle the non-Gaussian likelihood. The latent function values are represented by centered (whitened) variables, where:
        v ~ N(0, I)
        f = Lv + m(x)
        with
        L L^T = K_θ


# Constructors:
    GPMC(X, y, m, k, lik)
    GPMC(; m=MeanZero(), k=SE(0.0, 0.0), lik=Likelihood()) # observation-free constructor

# Arguments:
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `lik::likelihood`   : Likelihood function

# Returns:
* `gp::GPMC`          : Gaussian process object, fitted to the training data if provided
""" ->
type GPMC{T<:Real} <: GPBase 
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    lik::Likelihood         # Likelihood is Gaussian for GPMC regression
    
    # Observation data
    nobsv::Int              # Number of observations
    X::MatF64               # Input observations
    y::Vector{T}            # Output observations
    v::Vector{Float64}      # Vector of latent (whitened) variables - N(0,1)
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    μ::Vector{Float64} 
    cK::AbstractPDMat       # (k + exp(2*obsNoise))
    ll::Float64             # Log-likelihood of general GPMC model
    dll::Vector{Float64}    # Gradient of log-likelihood
    target::Float64         # Log-target (i.e. Log-posterior)
    dtarget::Vector{Float64}# Gradient of the log-target (i.e. grad log-posterior)

    
    function (::Type{GPMC{T}}){T<:Real}(X::MatF64, y::Vector{T}, m::Mean, k::Kernel, lik::Likelihood)
        dim, nobsv = size(X)
        v = zeros(nobsv)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new{T}(m, k, lik, nobsv, X, y, v, KernelData(k, X), dim)
        initialise_target!(gp)
        return gp
    end
end

GPMC{T<:Real}(X::Matrix{Float64}, y::Vector{T}, meanf::Mean, kernel::Kernel, lik::Likelihood) = GPMC{T}(X, y, meanf, kernel, lik)

# # Convenience constructor
GP{T<:Real}(X::Matrix{Float64}, y::Vector{T}, m::Mean, k::Kernel, lik::Likelihood) = GPMC{T}(X, y, m, k, lik)

# Creates GP object for 1D case
GPMC{T<:Real}(x::Vector{Float64}, y::Vector{T}, meanf::Mean, kernel::Kernel, lik::Likelihood) = GPMC{T}(x', y, meanf, kernel, lik)

GP{T<:Real}(x::Vector{Float64}, y::Vector{T}, m::Mean, k::Kernel, lik::Likelihood) = GPMC{T}(x', y, m, k, lik)

"""Initialise the log-likelihood function of a general GP model"""
function initialise_ll!(gp::GPMC)
    # log p(Y|v,θ) 
    gp.μ = mean(gp.m,gp.X)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + 1e-6*I)
    F = unwhiten(gp.cK,gp.v) + gp.μ 
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
end

# modification of initialise_ll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_ll!(gp::GPMC)
    Σbuffer = gp.cK.mat
    gp.μ = mean(gp.m,gp.X)
    cov!(Σbuffer, gp.k, gp.X, gp.data)
    Σbuffer += 1e-6*I
    chol_buffer = gp.cK.chol.factors
    copy!(chol_buffer, Σbuffer)
    chol = cholfact!(Symmetric(chol_buffer))
    gp.cK = PDMats.PDMat(Σbuffer, chol)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
end



# dlog p(Y|v,θ)
""" Update gradient of the log-likelihood dlog p(Y|v,θ) """
function update_ll_and_dll!(gp::GPMC, Kgrad::MatF64;
    lik::Bool=true,  # include gradient components for the likelihood parameters
    domean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    size(Kgrad) == (gp.nobsv, gp.nobsv) || throw(ArgumentError, 
    @sprintf("Buffer for Kgrad should be a %dx%d matrix, not %dx%d",
             gp.nobsv, gp.nobsv,
             size(Kgrad,1), size(Kgrad,2)))
    
    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    update_ll!(gp)
    Lv = unwhiten(gp.cK,gp.v)
    
    gp.dll = Array{Float64}(gp.nobsv + lik*n_lik_params + domean*n_mean_params + kern*n_kern_params)
    dl_df=dlog_dens_df(gp.lik, Lv + gp.μ, gp.y)

    U = triu(gp.cK.chol.factors)
    L = U'
    gp.dll[1:gp.nobsv] = L'dl_df
    
    i=gp.nobsv+1 
    if lik && n_lik_params > 0
        Lgrads = dlog_dens_dθ(gp.lik,Lv + gp.μ, gp.y)
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
        # L_bar = zeros(gp.nobsv, gp.nobsv)
        # tril!(LinAlg.BLAS.ger!(1.0, dl_df, gp.v, L_bar)) # grad_ll_L
        L_bar = tril(dl_df * gp.v')
        chol_unblocked_rev!(L, L_bar)
        for iparam in 1:n_kern_params
            grad_slice!(Kgrad, gp.k, gp.X, gp.data, iparam)
            gp.dll[i] = vecdot(Kgrad, L_bar)
            i+=1
        end
    end
end


""" GPMC: Initialise the target, which is assumed to be the log-posterior, log p(θ,v|y) ∝ log p(y|v,θ) + log p(v) +  log p(θ) """
function initialise_target!(gp::GPMC)
    initialise_ll!(gp)
    gp.target = gp.ll + sum(-0.5*gp.v.*gp.v-0.5*log(2*pi)) + prior_logpdf(gp.lik) + prior_logpdf(gp.m) + prior_logpdf(gp.k) 
end    

""" GPMC: Update the target, which is assumed to be the log-posterior, log p(θ,v|y) ∝ log p(y|v,θ) + log p(v) +  log p(θ) """
function update_target!(gp::GPMC)
    update_ll!(gp)
    gp.target = gp.ll + sum(-0.5*gp.v.*gp.v-0.5*log(2*pi)) + prior_logpdf(gp.lik) + prior_logpdf(gp.m) + prior_logpdf(gp.k) 
end    

""" GPMC: A function to update the target (aka log-posterior) and its derivative """
function update_target_and_dtarget!(gp::GPMC, Kgrad::MatF64; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    update_target!(gp)
    update_ll_and_dll!(gp, Kgrad; lik=lik, domean=domean, kern=kern)
    gp.dtarget = gp.dll + prior_gradlogpdf(gp; lik=lik, domean=domean, kern=kern)
end
""" GPMC: A function to update the target (aka log-posterior) and its derivative """
function update_target_and_dtarget!(gp::GPMC; kwargs...)
    Kgrad = Array{Float64}( gp.nobsv, gp.nobsv)
    update_target_and_dtarget!(gp, Kgrad; kwargs...)
end


#Calculate the mean and variance of predictive distribution p(y^*|x^*,D,θ) at test locations x^*
function predict_y{M<:MatF64}(gp::GPMC, x::M; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    return predict_obs(gp.lik, μ, σ2)
end

# 1D Case for prediction
predict_y{V<:VecF64}(gp::GPMC, x::V; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)


## compute predictions
function _predict{M<:MatF64}(gp::GPMC, X::M)
    n = size(X, 2)
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    fmu =  mean(gp.m,X) + Lck'gp.v     # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Add jitter to get stable covariance
    while true
        Sigma_raw = try
            PDMat(Sigma_raw)
            break
        catch
            PDMat(Sigma_raw+(1e-5*sum(diag(Sigma_raw))/n)*I)
        end
    end
    fSigma = PDMat(Sigma_raw)
    return fmu, fSigma
end


# Sample from functions from the GP 
function rand!{M<:MatF64}(gp::GPMC, X::M, A::DenseMatrix)
    nobsv = size(X,2)
    n_sample = size(A,2)

    if gp.nobsv == 0
        # Prior mean and covariance
        μ = mean(gp.m, X);
        Σraw = cov(gp.k, X);
        # Add jitter to get stable covariance
        while true
            Σraw = try
                PDMat(Σraw)
                break
            catch
                PDMat(Σraw+(1e8*sum(diag(Σraw))/nobsv)*I)
            end
        end
        Σ = Σraw
    else
        # Posterior mean and covariance
        μ, Σ = predict_f(gp, X; full_cov=true)
    end
    
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobsv, n_sample)))
end

#Samples random function values f
function rand{M<:MatF64}(gp::GPMC, X::M, n::Int)
    nobsv=size(X,2)
    A = Array{Float64}( nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GP
rand{V<:VecF64}(gp::GPMC, x::V, n::Int) = rand(gp, x', n)

# Generate only one sample from the GP and returns a vector
rand{M<:MatF64}(gp::GPMC,X::M) = vec(rand(gp,X,1))
rand{V<:VecF64}(gp::GPMC,x::V) = vec(rand(gp,x',1))


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

function set_params!(gp::GPMC, hyp::Vector{Float64}; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    gp.v = hyp[1:gp.nobsv]
    i=gp.nobsv+1  
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

function prior_gradlogpdf(gp::GPMC; lik::Bool=true, domean::Bool=true, kern::Bool=true)
    grad = -gp.v
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


function show(io::IO, gp::GPMC)
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
        show(io, round(gp.target,3))
    end
end

