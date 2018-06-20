import Base.show
# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process, with non-Gaussian observations, is defined in terms of its likelihood function, mean and covaiance (kernel) functions, which are user defined. We use a variational method to handle the non-Gaussian likelihood by approximating the latent function with a Gaussian approximation of the form, q(f) = N(Kα,[K⁻¹+diag(λ)]⁻¹), where Kα is a reparameterisation of the mean and K is positive semi-definite matrix. See Opper and Archambeau (2009), "The variational Gaussian approximation revisited" for full details.

# Constructors:
    GPV(X, y, m, k, lik)
    GPV(; m=MeanZero(), k=SE(0.0, 0.0), lik=Likelihood()) # observation-free constructor

# Arguments:
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `lik::likelihood`   : Likelihood function

# Returns:
* `gp::GPV`          : Gaussian process object, fitted to the training data if provided
""" ->
type GPV{T<:Real} <: GPBase 
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    lik::Likelihood         # Likelihood 
    
    # Observation data
    nobsv::Int              # Number of observations
    X::MatF64               # Input observations
    y::Vector{T}            # Output observations
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    μ::Vector{Float64}      
    cK::AbstractPDMat       # (k + exp(2*obsNoise))
    qμ::Vector{Float64}     # Mean of Gaussian variational approximation
    qΣ::AbstractPDMat       # Covariance matrix of the variational approximation
    ll::Float64             # Log-likelihood of general GPV model
    dll::Vector{Float64}    # Gradient of log-likelihood
    target::Float64         # Log-target (i.e. Log-posterior)
    dtarget::Vector{Float64}# Gradient of the log-target (i.e. grad log-posterior)

    
    function (::Type{GPV{T}}){T<:Real}(X::MatF64, y::Vector{T}, m::Mean, k::Kernel, lik::Likelihood)
        dim, nobsv = size(X)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new{T}(m, k, lik, nobsv, X, y, KernelData(k, X), dim)
        initialise_target!(gp)
        return gp
    end
end

GPV{T<:Real}(X::Matrix{Float64}, y::Vector{T}, meanf::Mean, kernel::Kernel, lik::Likelihood) = GPV{T}(X, y, meanf, kernel, lik)

# Creates GP object for 1D case
GPV{T<:Real}(x::Vector{Float64}, y::Vector{T}, meanf::Mean, kernel::Kernel, lik::Likelihood) = GPV{T}(x', y, meanf, kernel, lik)


"""Initialise the variational lower bound on the log-likelihood function of a general GP model"""
function initialise_ll!(gp::GPV)
    gp.qμ = zeros(gp.nobsv)
    gp.qΣ = eye(gp.nobsv)

    kl = 0.5(dot(gp.qμ,gp.qμ) - logdet(gp.qΣ) + sum(diag(gp.qΣ).^2)) #KL prior gives the divergence between two Gaussians

    gp.μ = mean(gp.m,gp.X)          #mean function 
    Σ = cov(gp.k, gp.X, gp.data)    #kernel function
    gp.cK = PDMat(Σ + 1e-6*I)       
    Fmean = unwhiten(gp.cK, gp.qμ) + gp.μ      # K⁻¹q_μ 
    Fvar = unwhiten(gp.cK, gp.qΣ)              # K⁻¹q_Σ
    varExp = var_exp(gp.lik, Fmean, Fvar)      # ∫log p(y|f)q(f), where q(f) is a Gaussian approx.
    gp.ll = varExp - KL                                 # Log-likelihood lower bound
end

"""
Update the covariance matrix and its Cholesky decomposition.
"""
function update_cK!(gp::GPV)
    old_cK = gp.cK
    Σbuffer = old_cK.mat
    cov!(Σbuffer, gp.k, gp.X, gp.data)
    for i in 1:gp.nobsv
        Σbuffer[i,i] += 1e-6 # no logNoise for GPV
    end
    chol_buffer = old_cK.chol.factors
    copy!(chol_buffer, Σbuffer)
    chol = cholfact!(Symmetric(chol_buffer))
    gp.cK = PDMats.PDMat(Σbuffer, chol)
end

# modification of initialise_ll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_ll!(gp::GPV; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    if kern
        # only need to update the covariance matrix
        # if the covariance parameters have changed
        update_cK!(gp)
    end
    gp.μ = mean(gp.m,gp.X)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
end

""" Update gradient of the log-likelihood dlog p(Y|v,θ) """
function update_dll!(gp::GPV, Kgrad::MatF64, L_bar::MatF64;
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

    gp.dll = Array{Float64}(process*gp.nobsv + lik*n_lik_params + domean*n_mean_params + kern*n_kern_params)

    Lv = unwhiten(gp.cK, gp.v)
    dl_df = dlog_dens_df(gp.lik, Lv + gp.μ, gp.y)
    i = 1
    if process
        A_mul_B!(view(gp.dll, i:i+gp.nobsv-1), gp.cK.chol[:U], dl_df)
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
        L_bar[:] = 0.0
        LinAlg.BLAS.ger!(1.0, dl_df, gp.v, L_bar)
        tril!(L_bar)
        # ToDo:
        # the following two steps allocates memory
        # and are fickle, reaching into the internal
        # implementation of the cholesky decomposition
        L = gp.cK.chol[:L].data
        tril!(L)
        #
        chol_unblocked_rev!(L, L_bar)
        for iparam in 1:n_kern_params
            grad_slice!(Kgrad, gp.k, gp.X, gp.data, iparam)
            gp.dll[i] = vecdot(Kgrad, L_bar)
            i+=1
        end
    end
end

function update_ll_and_dll!(gp::GPV, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_ll!(gp; kwargs...)
    update_dll!(gp, Kgrad, L_bar; kwargs...)
end


""" GPV: Initialise the target, which is assumed to be the log-posterior, log p(θ|y) ∝ log p(y|θ) +  log p(θ) """
function initialise_target!(gp::GPV)
    initialise_ll!(gp)
    gp.target = gp.ll + prior_logpdf(gp.lik) + prior_logpdf(gp.m) + prior_logpdf(gp.k) 
end    

""" GPV: Update the target, which is assumed to be the log-posterior, log p(θ|y) ∝ log p(y|θ) +  log p(θ) """
function update_target!(gp::GPV; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
    update_ll!(gp; process=process, lik=lik, domean=domean, kern=kern)
    gp.target = gp.ll + prior_logpdf(gp.lik) + prior_logpdf(gp.m) + prior_logpdf(gp.k) 
end    

function update_dtarget!(gp::GPV, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_dll!(gp, Kgrad, L_bar; kwargs...)
    gp.dtarget = gp.dll + prior_gradlogpdf(gp; kwargs...)
end

""" GPV: A function to update the target (aka log-posterior) and its derivative """
function update_target_and_dtarget!(gp::GPV, Kgrad::MatF64, L_bar::MatF64; kwargs...)
    update_target!(gp; kwargs...)
    update_dtarget!(gp, Kgrad, L_bar; kwargs...)
end
""" GPV: A function to update the target (aka log-posterior) and its derivative """
function update_target_and_dtarget!(gp::GPV; kwargs...)
    Kgrad = Array{Float64}(gp.nobsv, gp.nobsv)
    L_bar = Array{Float64}(gp.nobsv, gp.nobsv)
    update_target_and_dtarget!(gp, Kgrad, L_bar; kwargs...)
end


#Calculate the mean and variance of predictive distribution p(y^*|x^*,D,θ) at test locations x^*
function predict_y{M<:MatF64}(gp::GPV, x::M; full_cov::Bool=false)
    μ, σ2 = predict_f(gp, x; full_cov=full_cov)
    return predict_obs(gp.lik, μ, σ2)
end

# 1D Case for prediction
predict_y{V<:VecF64}(gp::GPV, x::V; full_cov::Bool=false) = predict_y(gp, x'; full_cov=full_cov)


## compute predictions
function _predict{M<:MatF64}(gp::GPV, X::M)
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
function rand!{M<:MatF64}(gp::GPV, X::M, A::DenseMatrix)
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
function rand{M<:MatF64}(gp::GPV, X::M, n::Int)
    nobsv=size(X,2)
    A = Array{Float64}( nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GP
rand{V<:VecF64}(gp::GPV, x::V, n::Int) = rand(gp, x', n)

# Generate only one sample from the GP and returns a vector
rand{M<:MatF64}(gp::GPV,X::M) = vec(rand(gp,X,1))
rand{V<:VecF64}(gp::GPV,x::V) = vec(rand(gp,x',1))


function get_params(gp::GPV; lik::Bool=true, domean::Bool=true, kern::Bool=true)
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

function set_params!(gp::GPV, hyp::Vector{Float64}; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
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

function prior_gradlogpdf(gp::GPV; process::Bool=true, lik::Bool=true, domean::Bool=true, kern::Bool=true)
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


function show(io::IO, gp::GPV)
    println(io, "GP Variational object:")
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
        print(io,"\n  Variational lower bound = ")
        show(io, round(gp.target,3))
    end
end

