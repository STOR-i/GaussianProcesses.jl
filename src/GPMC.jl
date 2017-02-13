import Base.show
# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its likelihood function and mean and covaiance (kernel) functions, which are user defined. We use a Monte Carlo methods to handle the non-Gaussian likelihood. The latent function values are represented by centered (whitened) variables, where:
        v ~ N(0, I)
        f = Lv + m(x)
        with
        L L^T = K_θ


# Constructors:
    GP(X, y, m, k, lik)
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
type GPMC{T<:Real}
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    lik::Likelihood         # Likelihood is Gaussian for GPMC regression
    
    # Observation data
    nobsv::Int              # Number of observations
    X::Matrix{Float64}      # Input observations
    y::Vector{T}            # Output observations
    v::Vector{Float64}      # Vector of latent (whitened) variables - N(0,1)
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    μ::Vector{Float64} 
    cK::AbstractPDMat       # (k + exp(2*obsNoise))
    ll::Float64             # Log-likelihood of general GPMC model
    dll::Vector{Float64}    # Gradient of log-likelihood
    lp::Float64             # Log-posterior of general GPMC model
    dlp::Vector{Float64}    # Gradient of log-posterior
    
    function GPMC{S<:Real}(X::Matrix{Float64}, y::Vector{S}, m::Mean, k::Kernel, lik::Likelihood)
        dim, nobsv = size(X)
        v = zeros(nobsv)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(m, k, lik, nobsv, X, y, v, KernelData(k, X), dim)
        initialise_lpost!(gp)
        return gp
    end
end

# Creates GP object for 1D case
GPMC(x::Vector{Float64}, y::Vector, meanf::Mean, kernel::Kernel, lik::Likelihood) = GPMC(x', y, meanf, kernel, lik)



#initiate the log-likelihood function of a general GP model
function initialise_ll!(gp::GPMC)
    # log p(Y|v,θ) 
    gp.μ = mean(gp.m,gp.X)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + 1e-6*eye(gp.nobsv))
    F = unwhiten(gp.cK,gp.v) + gp.μ 
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
end

# modification of initialise_ll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_ll!(gp::GPMC)
    Σbuffer = gp.cK.mat
    gp.μ = mean(gp.m,gp.X)
    cov!(Σbuffer, gp.k, gp.X, gp.data)
    for i in 1:gp.nobsv
        Σbuffer[i,i] += 1e-8
    end
    chol_buffer = gp.cK.chol.factors
    copy!(chol_buffer, Σbuffer)
    chol = cholfact!(Symmetric(chol_buffer))
    gp.cK = PDMats.PDMat(Σbuffer, chol)
    F = unwhiten(gp.cK,gp.v) + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y)) #Log-likelihood
end



# dlog p(Y|v,θ)
""" Update gradient of the log-likelihood """
function update_ll_and_dll!(gp::GPMC, Kgrad::MatF64;
    lik::Bool=true,  # include gradient components for the likelihood parameters
    mean::Bool=true, # include gradient components for the mean parameters
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
    
    gp.dll = Array(Float64,gp.nobsv + lik*n_lik_params + mean*n_mean_params + kern*n_kern_params)
    dl_df=dlog_dens_df(gp.lik, Lv + gp.μ, gp.y)

    U = triu(gp.cK.chol.factors)
    L = U'
    gp.dll[1:gp.nobsv] = L'dl_df
    
    i=gp.nobsv+1 
    if lik  && n_lik_params>0
        Lgrads = dlog_dens_dθ(gp.lik,Lv + gp.μ, gp.y)
        for j in 1:n_lik_params
            gp.dll[i] = sum(Lgrads[:,j])
            i += 1
        end
    end
    if mean && n_mean_params>0
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



#log p(θ,v|y) ∝ log p(y|v,θ) + log p(v) +  log p(θ)
function initialise_lpost!(gp::GPMC)
    initialise_ll!(gp)
    prior = 0.0 
    if num_params(gp.lik)>0
        prior = prior_logpdf(gp.lik)
    end    
    gp.lp = gp.ll + sum(-0.5*gp.v.*gp.v-0.5*log(2*pi)) + prior + prior_logpdf(gp.k) 
end    

#log p(θ,v|y) ∝ log p(y|v,θ) + log p(v) +  log p(θ)
function update_lpost!(gp::GPMC)
    update_ll!(gp)
    prior = 0.0 
    if num_params(gp.lik)>0
        prior = prior_logpdf(gp.lik)
    end    
    gp.lp = gp.ll + sum(-0.5*gp.v.*gp.v-0.5*log(2*pi)) + prior + prior_logpdf(gp.k) 
end    

#dlog p(θ,v|y) ∝ dlog p(y|v,θ) + dlog p(v) +  dlog p(θ)
function update_lpost_and_dlpost!(gp::GPMC, Kgrad::MatF64; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    update_ll_and_dll!(gp::GPMC, Kgrad; lik=lik, mean=mean, kern=kern)
    if num_params(gp.lik)>0
        gp.dlp = gp.dll + [-gp.v;prior_gradlogpdf(gp.lik);zeros(num_params(gp.m));prior_gradlogpdf(gp.k)] 
    else
        gp.dlp = gp.dll + [-gp.v;zeros(num_params(gp.m));prior_gradlogpdf(gp.k)] 
    end

end    


@doc """
    # Description
    Calculates the posterior mean and variance of Gaussian Process at specified points

    # Arguments:
    * `gp::GP`: Gaussian Process object
    * `X::Matrix{Float64}`:  matrix of points for which one would would like to predict the value of the latent process, f.
                           (each column of the matrix is a point)
    * `obs::Bool`: Returns predicted observations 'y' calculated at test points X, set to 'false' by default    
    # Returns:
    * `(mu, Sigma)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
                                                        process at the specified points
    """ ->
function predict{M<:MatF64}(gp::GPMC, x::M; obs::Bool=false, full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        μ, σ2 = _predict(gp, x)
    else
        ## Calculate prediction for each point independently
            μ = Array(Float64, size(x,2))
            σ2 = similar(μ)
        for k in 1:size(x,2)
            m, sig = _predict(gp, x[:,k:k])
            μ[k] = m[1]
            σ2[k] = max(full(sig)[1,1], 0.0)
        end
    end
    if obs #need to handle full_cov case
       μ, σ2 = predict_obs(gp.lik, μ, σ2)
    end
    return μ, σ2
end

# 1D Case for prediction
predict{V<:VecF64}(gp::GPMC, x::V; obs::Bool=false, full_cov::Bool=false) = predict(gp, x'; obs=obs, full_cov=full_cov)

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
            PDMat(Sigma_raw+1e-8*sum(diag(Sigma_raw))/n*eye(n))
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
                PDMat(Σraw+1e-8*sum(diag(Σraw))/nobsv*eye(nobsv))
            end
        end
        Σ = Σraw
    else
        # Posterior mean and covariance
        μ, Σ = predict(gp, X; full_cov=true)
    end
    
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobsv, n_sample)))
end

#Samples random function values f
function rand{M<:MatF64}(gp::GPMC, X::M, n::Int)
    nobsv=size(X,2)
    A = Array(Float64, nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GP
rand{V<:VecF64}(gp::GPMC, x::V, n::Int) = rand(gp, x', n)

# Generate only one sample from the GP and returns a vector
rand{M<:MatF64}(gp::GPMC,X::M) = vec(rand(gp,X,1))
rand{V<:VecF64}(gp::GPMC,x::V) = vec(rand(gp,x',1))


function get_params(gp::GPMC; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    append!(params, gp.v)
    if lik  && num_params(gp.lik)>0
        append!(params, get_params(gp.lik))
    end
    if mean && num_params(gp.m)>0
        append!(params, get_params(gp.m))
    end
    if kern
        append!(params, get_params(gp.k))
    end
    return params
end

function set_params!(gp::GPMC, hyp::Vector{Float64}; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    gp.v = hyp[1:gp.nobsv]
    i=gp.nobsv+1  
    if lik  && n_lik_params>0
        set_params!(gp.lik, hyp[i:i+n_lik_params-1]);
        i += n_lik_params
    end
    if mean && n_mean_params>0
        set_params!(gp.m, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end
    if kern
        set_params!(gp.k, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
end
    
function push!(gp::GPMC, X::Matrix{Float64}, y::Vector{Float64})
    warn("push! method is currently inefficient as it refits all observations")
    if gp.nobsv == 0
        GaussianProcesses.fit!(gp, X, y)
    elseif size(X,1) != size(gp.X,1)
        error("New input observations must have dimensions consistent with existing observations")
    else
        GaussianProcesses.fit!(gp, cat(2, gp.X, X), cat(1, gp.y, y))
    end
end

push!(gp::GPMC, x::Vector{Float64}, y::Vector{Float64}) = push!(gp, x', y)
push!(gp::GPMC, x::Float64, y::Float64) = push!(gp, [x], [y])
push!(gp::GPMC, x::Vector{Float64}, y::Float64) = push!(gp, reshape(x, length(x), 1), [y])

function show(io::IO, gp::GPMC)
    println(io, "GP object:")
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
        show(io, round(gp.lp,3))
    end
end


