import Base.show
# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.

# Constructors:
    GP(X, y, m, k, logNoise)
    GPMC(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8) # observation-free constructor

# Arguments:
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `logNoise::Float64` : Log of the standard deviation for the observation noise. The default is -1e8, which is equivalent to assuming no observation noise.

# Returns:
* `gp::GPMC`            : Gaussian process object, fitted to the training data if provided
""" ->
type GPMC{T<:Real}
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    lik::Likelihood         # Likelihood is Gaussian for GPMC regression
    # logNoise::Float64       # log standard deviation of observation noise
    
    # Observation data
    nobsv::Int              # Number of observations
    X::Matrix{Float64}      # Input observations
    y::Vector{T}            # Output observations
    v::Vector{Float64}      # Vector of latent (whitened) variables - N(0,1)
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    cK::AbstractPDMat       # (k + exp(2*obsNoise))
    alpha::Vector{Float64}  # (k + exp(2*obsNoise))⁻¹y
    ll::Float64             # Log-likelihood of general GPMC model
    

    function GPMC{S<:Real}(X::Matrix{Float64}, y::Vector{S}, m::Mean, k::Kernel, lik::Likelihood)
        dim, nobsv = size(X)
        v = zeros(nobsv)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        #=
        This is a vanilla implementation of a GPMC with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, where
        v ~ N(0, I)
        f = Lv + m(x)
        with
        L L^T = K
        =#
        gp = new(m, k, lik, nobsv, X, y, v, KernelData(k, X), dim)
        likelihood!(gp)
        return gp
    end
end

# Creates GP object for 1D case
GPMC(x::Vector{Float64}, y::Vector, meanf::Mean, kernel::Kernel, lik::Likelihood) = GPMC(x', y, meanf, kernel, lik)

@doc """
# Description
Fits an existing Gaussian process to a set of training points.

# Arguments:
* `gp::GP`: Exiting Gaussian process object
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations

# Returns:
* `gp::GP`            : A Gaussian process fitted to the training data
""" ->
function fit!(gp::GPMC, X::Matrix{Float64}, y::Vector{Float64})
    length(y) == size(X,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.X = X
    gp.y = y
    gp.data = KernelData(gp.k, X)
    gp.dim, gp.nobsv = size(X)
    initialise_mll!(gp)
    return gp
end

fit!(gp::GPMC, x::Vector{Float64}, y::Vector{Float64}) = fit!(gp, x', y)


#Likelihood function of general GP model
function likelihood!(gp::GPMC)
    # log p(Y|v,θ) 
    μ = mean(gp.m,gp.X)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + 1e-8*eye(gp.nobsv))
    F = gp.cK*gp.v + μ
    gp.ll = sum(logdens(gp.lik,F,gp.y))
end

function conditional(gp::GPMC, X::Matrix{Float64})
    n = size(X, 2)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + 1e-6*eye(gp.nobsv))
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Hack to get stable covariance
    fSigma = try PDMat(Sigma_raw) catch; PDMat(Sigma_raw+1e-8*sum(diag(Sigma_raw))/n*eye(n)) end
    fmu = mean(gp.m,X) + Lck*gp.v        # Predictive mean
    return fmu, fSigma
end

@doc """
    # Description
    Calculates the posterior mean and variance of Gaussian Process at specified points

    # Arguments:
    * `gp::GP`: Gaussian Process object
    * `X::Matrix{Float64}`:  matrix of points for which one would would like to predict the value of the process.
                           (each column of the matrix is a point)

    # Keyword Arguments
    * `full_cov::Bool`: indicates whether full covariance matrix should be returned instead of only variances (default is false)

    # Returns:
    * `(mu, Sigma)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
                                                        process at the specified points
    """ ->
function predict(gp::GPMC, X::Matrix{Float64}; full_cov::Bool=false)
    size(X,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if typeof(gp.lik)!=Gaussian
        return μ, σ2 = conditional(gp, X)
    else
        if full_cov
            return _predict(gp, X)
        else
            ## Calculate prediction for each point independently
            μ = Array(Float64, size(X,2))
            σ2 = similar(μ)
            for k in 1:size(X,2)
                m, sig = _predict(gp, X[:,k:k])
                μ[k] = m[1]
                σ2[k] = max(full(sig)[1,1], 0.0)
            end
            return μ, σ2
        end
    end
end

# 1D Case for prediction
predict(gp::GPMC, x::Vector{Float64}; full_cov::Bool=false) = predict(gp, x'; full_cov=full_cov)

## compute predictions
function _predict(gp::GPMC, X::Array{Float64})
    n = size(X, 2)
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    mu = mean(gp.m,X) + cK*gp.alpha        # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Hack to get stable covariance
    Sigma = try PDMat(Sigma_raw) catch; PDMat(Sigma_raw+1e-8*sum(diag(Sigma_raw))/n*eye(n)) end 
    return (mu, Sigma)
end

function get_params(gp::GPMC; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    if lik  && num_params(gp.lik)>0; push!(params, get_params(gp.lik)); end
    if mean;  append!(params, get_params(gp.m)); end
    if kern; append!(params, get_params(gp.k)); end
    return params
end

function set_params!(gp::GPMC, hyp::Vector{Float64}; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    # println("mean=$(mean)")
    gp.v = hyp[1:gp.nobsv]
    if lik  && num_params(gp.lik)>0;; set_params!(gp.lik, hyp[gp.nobsv+1:gp.nobsv+num_params(gp.lik)]); end
    if mean; set_params!(gp.m, hyp[gp.nobsv+1+num_params(gp.lik):gp.nobsv+num_params(gp.lik)+num_params(gp.m)]); end
    if kern; set_params!(gp.k, hyp[end-num_params(gp.k)+1:end]); end
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
    if (gp.nobsv == 0)
        println("  No observation data")
    else
        println(io, "  Input observations = ")
        show(io, gp.X)
        print(io,"\n  Output observations = ")
        show(io, gp.y)
        if typeof(gp.lik)!=Gaussian
            print(io,"\n  Log-Likelihood = ")
            show(io, round(gp.ll,3))
        else
            print(io,"\n  Marginal Log-Likelihood = ")
            show(io, round(gp.mLL,3))

        end            
    end
end

