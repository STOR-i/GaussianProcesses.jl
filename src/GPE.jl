import Base.show
# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.

# Constructors:
    GPE(X, y, m, k, logNoise)
    GPE(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8) # observation-free constructor

# Arguments:
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `logNoise::Float64` : Log of the standard deviation for the observation noise. The default is -1e8, which is equivalent to assuming no observation noise.

# Returns:
* `gp::GPE`            : Gaussian process object, fitted to the training data if provided
""" ->
type GPE <: GP
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    logNoise::Float64       # log standard deviation of observation noise
    
    # Observation data
    nobsv::Int              # Number of observations
    X::Matrix{Float64}      # Input observations
    y::Vector{Float64}      # Output observations
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    cK::AbstractPDMat       # (k + exp(2*obsNoise))
    alpha::Vector{Float64}  # (k + exp(2*obsNoise))⁻¹y
    mLL::Float64            # Marginal log-likelihood
    dmLL::Vector{Float64}   # Gradient marginal log-likelihood
    
    function GPE(X::Matrix{Float64}, y::Vector{Float64}, m::Mean, k::Kernel, logNoise::Float64=-1e8)
        dim, nobsv = size(X)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(m, k, logNoise, nobsv, X, y, KernelData(k, X), dim)
        initialise_target!(gp)
        return gp
    end
    
    GPE(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8) =  new(m, k, logNoise, 0)
    
end

# Creates GPE object for 1D case
GPE(x::Vector{Float64}, y::Vector{Float64}, meanf::Mean, kernel::Kernel, logNoise::Float64=-1e8) = GPE(x', y, meanf, kernel, logNoise)

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
function fit!(gp::GPE, X::Matrix{Float64}, y::Vector{Float64})
    length(y) == size(X,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.X = X
    gp.y = y
    gp.data = KernelData(gp.k, X)
    gp.dim, gp.nobsv = size(X)
    initialise_target!(gp)
    return gp
end

fit!(gp::GPE, x::Vector{Float64}, y::Vector{Float64}) = fit!(gp, x', y)


# Update auxiliarly data in GPE object after changes have been made
function initialise_mll!(gp::GPE)
    μ = mean(gp.m,gp.X)
    Σ = cov(gp.k, gp.X, gp.data)
    for i in 1:gp.nobsv
        ## add observation noise
        Σ[i,i] += exp(2*gp.logNoise) + 1e-8
    end
    gp.cK = PDMat(Σ)
    gp.alpha = gp.cK \ (gp.y - μ)
    gp.mLL = -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

function initialise_target!(gp::GPE)
    initialise_mll!(gp)
end    

# modification of initialise_mll! that reuses existing matrices to avoid
# unnecessary memory allocations, which speeds things up significantly
function update_mll!(gp::GPE)
    Σbuffer = gp.cK.mat
    μ = mean(gp.m,gp.X)
    cov!(Σbuffer, gp.k, gp.X, gp.data)
    for i in 1:gp.nobsv
        Σbuffer[i,i] += exp(2*gp.logNoise) + 1e-8
    end
    chol_buffer = gp.cK.chol.factors
    copy!(chol_buffer, Σbuffer)
    chol = cholfact!(Symmetric(chol_buffer))
    gp.cK = PDMats.PDMat(Σbuffer, chol)
    gp.alpha = gp.cK \ (gp.y - μ)
    gp.mLL = -dot((gp.y - μ),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end


#function to udpate the target
function update_target!(gp::GPE)
    update_mll!(gp)
end    

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
    size(ααinvcKI) == (nobsv, nobsv) || throw(ArgumentError, 
                @sprintf("Buffer for ααinvcKI should be a %dx%d matrix, not %dx%d",
                         nobsv, nobsv,
                         size(ααinvcKI,1), size(ααinvcKI,2)))
    ααinvcKI[:,:] = 0.0
    @inbounds for i in 1:nobsv
        ααinvcKI[i,i] = -1.0
    end
    A_ldiv_B!(cK.chol, ααinvcKI)
    LinAlg.BLAS.ger!(1.0, α, α, ααinvcKI)
end

""" Update gradient of marginal log-likelihood """
function update_mll_and_dmll!(gp::GPE,
    Kgrad::MatF64,
    ααinvcKI::MatF64
    ; 
    noise::Bool=true, # include gradient component for the logNoise term
    mean::Bool=true, # include gradient components for the mean parameters
    kern::Bool=true, # include gradient components for the spatial kernel parameters
    )
    size(Kgrad) == (gp.nobsv, gp.nobsv) || throw(ArgumentError, 
                @sprintf("Buffer for Kgrad should be a %dx%d matrix, not %dx%d",
                         gp.nobsv, gp.nobsv,
                         size(Kgrad,1), size(Kgrad,2)))
    update_mll!(gp)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)
    gp.dmLL = Array(Float64, noise + mean*n_mean_params + kern*n_kern_params)
    logNoise = gp.logNoise
    get_ααinvcKI!(ααinvcKI, gp.cK, gp.alpha)
    
    i=1
    if noise
        gp.dmLL[i] = exp(2.0*logNoise)*trace(ααinvcKI)
        i+=1
    end

    if mean
        Mgrads = grad_stack(gp.m, gp.X)
        for j in 1:n_mean_params
            gp.dmLL[i] = dot(Mgrads[:,j],gp.alpha)
            i += 1
        end
    end
    if kern
        for iparam in 1:n_kern_params
            grad_slice!(Kgrad, gp.k, gp.X, gp.data, iparam)
            gp.dmLL[i] = vecdot(Kgrad,ααinvcKI)/2.0
            i+=1
        end
    end
end

#function to update the marginal log-likelihood and its derivative
function update_target_and_dtarget!(gp::GPE; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    noise = lik
    Kgrad = Array(Float64, gp.nobsv, gp.nobsv)
    ααinvcKI = Array(Float64, gp.nobsv, gp.nobsv)
    update_mll_and_dmll!(gp, Kgrad, ααinvcKI, noise=noise,mean=mean,kern=kern)
end

@doc """
# Description
Calculates the posterior mean and variance of the Gaussian Process function at specified points

# Arguments:
* `gp::GPE`: Gaussian Process object
* `X::Matrix{Float64}`:  matrix of points for which one would would like to predict the value of the process.
                       (each column of the matrix is a point)

# Keyword Arguments
* `full_cov::Bool`: indicates whether full covariance matrix should be returned instead of only variances (default is false)

# Returns:
* `(μ, σ²)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
                                                    process at the specified points
""" ->
function predictF{M<:MatF64}(gp::GPE, x::M; full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        return _predict(gp, x)
    else
        ## Calculate prediction for each point independently
            μ = Array(Float64, size(x,2))
            σ2 = similar(μ)
        for k in 1:size(x,2)
            m, sig = _predict(gp, x[:,k:k])
            μ[k] = m[1]
            σ2[k] = max(full(sig)[1,1], 0.0)
        end
        return μ, σ2
    end
end

# 1D Case for prediction
predictF{V<:VecF64}(gp::GPE, x::V; full_cov::Bool=false) = predictF(gp, x'; full_cov=full_cov)

## compute predictions
function _predict{M<:MatF64}(gp::GPE, X::M)
    n = size(X, 2)
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    mu = mean(gp.m,X) + cK*gp.alpha        # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Hack to get stable covariance
    Sigma = try PDMat(Sigma_raw) catch; PDMat(Sigma_raw+1e-8*sum(diag(Sigma_raw))/n*eye(n)) end 
    return (mu, Sigma)
end


# Sample from the GPE 
function rand!{M<:MatF64}(gp::GPE, X::M, A::DenseMatrix)
    nobsv = size(X,2)
    n_sample = size(A,2)

    if gp.nobsv == 0
        # Prior mean and covariance
        μ = mean(gp.m, X);
        Σraw = cov(gp.k, X);
        Σ = try PDMat(Σraw) catch; PDMat(Σraw+1e-8*sum(diag(Σraw))/nobsv*eye(nobsv)) end  
    else
        # Posterior mean and covariance
        μ, Σ = predictF(gp, X; full_cov=true)
    end
    
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobsv, n_sample)))
end

function rand{M<:MatF64}(gp::GPE, X::M, n::Int)
    nobsv=size(X,2)
    A = Array(Float64, nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GPE
rand{V<:VecF64}(gp::GPE, x::V, n::Int) = rand(gp, x', n)

# Generate only one sample from the GPE and returns a vector
rand{M<:MatF64}(gp::GPE,X::M) = vec(rand(gp,X,1))
rand{V<:VecF64}(gp::GPE,x::V) = vec(rand(gp,x',1))


function get_params(gp::GPE; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gp.logNoise); end
    if mean;  append!(params, get_params(gp.m)); end
    if kern; append!(params, get_params(gp.k)); end
    return params
end

function set_params!(gp::GPE, hyp::Vector{Float64}; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    if noise; gp.logNoise = hyp[1]; end
    i=1  
    if mean && n_mean_params>0
        set_params!(gp.m, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end
    if kern
        set_params!(gp.k, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
end
    
function push!(gp::GPE, X::Matrix{Float64}, y::Vector{Float64})
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

function show(io::IO, gp::GPE)
    println(io, "GPE object:")
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
        show(io, round(gp.mLL,3))
    end
end
