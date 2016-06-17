import Base.show

# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.

# Constructors:
    GP(x, y, m, k, logNoise)
    GP(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8) # observation-free constructor

# Arguments:
* `x::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `logNoise::Float64` : Log of the observation noise. The default is -1e8, which is equivalent to assuming no observation noise.

# Returns:
* `gp::GP`            : Gaussian process object, fitted to the training data if provided
""" ->
type GP
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    logNoise::Float64       # log variance of observation noise
    
    # Observation data
    nobsv::Int              # Number of observations
    X::Matrix{Float64}      # Input observations
    y::Vector{Float64}      # Output observations
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    cK::AbstractPDMat       # (k + obsNoise)
    alpha::Vector{Float64}  # (k + obsNoise)⁻¹y
    mLL::Float64            # Marginal log-likelihood
    dmLL::Vector{Float64}   # Gradient marginal log-likelihood
    
    function GP(X::Matrix{Float64}, y::Vector{Float64}, m::Mean, k::Kernel, logNoise::Float64=-1e8)
        dim, nobsv = size(X)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(m, k, logNoise, nobsv, X, y, KernelData(k, X), dim)
        update_mll!(gp)
        return gp
    end
    
    GP(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8) =  new(m, k, logNoise, 0)
    
end

# Creates GP object for 1D case
GP(x::Vector{Float64}, y::Vector{Float64}, meanf::Mean, kernel::Kernel, logNoise::Float64=-1e8) = GP(x', y, meanf, kernel, logNoise)

@doc """
# Description
Fits an existing Gaussian process to a set of training points.

# Arguments:
* `gp::GP`: Exiting Gaussian process object
* `x::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations

# Returns:
* `gp::GP`            : A Gaussian process fitted to the training data
""" ->
function fit!(gp::GP, X::Matrix{Float64}, y::Vector{Float64})
    length(y) == size(X,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.X = X
    gp.y = y
    gp.data = KernelData(X)
    gp.dim, gp.nobsv = size(X)
    update_mll!(gp)
    return gp
end

fit!(gp::GP, x::Vector{Float64}, y::Vector{Float64}) = fit!(gp, x', y)


# Update auxiliarly data in GP object after changes have been made
function update_mll!(gp::GP)
    m = mean(gp.m,gp.X)
    gp.cK = PDMat(cov(gp.k, gp.data) + exp(2*gp.logNoise)*eye(gp.nobsv) + 1e-8*eye(gp.nobsv))
    gp.alpha = gp.cK \ (gp.y - m)
    gp.mLL = -dot((gp.y-m),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 # Marginal log-likelihood
end

# Update gradient of marginal log likelihood
function update_mll_and_dmll!(gp::GP; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    update_mll!(gp::GP)
    gp.dmLL = Array(Float64, noise + mean*num_params(gp.m) + kern*num_params(gp.k))

    # Calculate Gradient with respect to hyperparameters

    #Derivative wrt the observation noise
    if noise
        #gp.dmLL[1] = exp(2*gp.logNoise)*trace((gp.alpha*gp.alpha' - gp.L'\(gp.L\eye(gp.nobsv))))
        gp.dmLL[1] = exp(2*gp.logNoise)*trace((gp.alpha*gp.alpha' - gp.cK \ eye(gp.nobsv)))
    end

    #Derivative wrt to mean hyperparameters, need to loop over as same with kernel hyperparameters
    if mean
        Mgrads = grad_stack(gp.x, gp.m)
        for i in 1:num_params(gp.m)
            gp.dmLL[i+noise] = -dot(Mgrads[:,i],gp.alpha)
        end
    end

    # Derivative of marginal log-likelihood with respect to kernel hyperparameters
    if kern
        Kgrads = grad_stack(gp.x, gp.k)   # [dK/dθᵢ]
        for i in 1:num_params(gp.k)
            gp.dmLL[i+mean*num_params(gp.m)+noise] = trace((gp.alpha*gp.alpha' - gp.cK \ eye(gp.nobsv))*Kgrads[:,:,i])/2
        end
    end
end


@doc """
# Description
Calculates the posterior mean and variance of Gaussian Process at specified points

# Arguments:
* `gp::GP`: Gaussian Process object
* `x::Matrix{Float64}`:  matrix of points for which one would would like to predict the value of the process.
                       (each column of the matrix is a point)

# Keyword Arguments
* `full_cov::Bool`: indicates whether full covariance matrix should be returned instead of only variances (default is false)

# Returns:
* `(mu, Sigma)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
                                                    process at the specified points
""" ->
function predict(gp::GP, x::Matrix{Float64}; full_cov::Bool=false)
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
predict(gp::GP, x::Vector{Float64}; full_cov::Bool=false) = predict(gp, x'; full_cov=full_cov)

## compute predictions
function _predict(gp::GP, X::Array{Float64})
    n = size(X, 2)
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    mu = mean(gp.m,X) + cK*gp.alpha        # Predictive mean
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Hack to get stable covariance
    Sigma = try PDMat(Sigma_raw) catch; PDMat(Sigma_raw+1e-8*sum(diag(Sigma_raw))/n*eye(n)) end 
    return (mu, Sigma)
end


# Sample from the GP 
function rand!(gp::GP, X::Matrix{Float64}, A::DenseMatrix)
    nobsv = size(X,2)
    n_sample = size(A,2)

    if gp.nobsv == 0
        # Prior mean and covariance
        μ = mean(gp.m, X);
        Σraw = cov(gp.k, X);
        Σ = try PDMat(Σraw) catch; PDMat(Σraw+1e-8*sum(diag(Σraw))/nobsv*eye(nobsv)) end  
    else
        # Posterior mean and covariance
        μ, Σ = predict(gp, X; full_cov=true)
    end
    
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobsv, n_sample)))
end

function rand(gp::GP, X::Matrix{Float64}, n::Int)
    nobsv=size(X,2)
    A = Array(Float64, nobsv, n)
    return rand!(gp, X, A)
end

# Sample from 1D GP
rand(gp::GP, x::Vector{Float64}, n::Int) = rand(gp, x', n)

# Generate only one sample from the GP and returns a vector
rand(gp::GP,X::Matrix{Float64}) = vec(rand(gp,X,1))
rand(gp::GP,x::Vector{Float64}) = vec(rand(gp,x',1))


function get_params(gp::GP; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gp.logNoise); end
    if mean;  append!(params, get_params(gp.m)); end
    if kern; append!(params, get_params(gp.k)); end
    return params
end

function set_params!(gp::GP, hyp::Vector{Float64}; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    # println("mean=$(mean)")
    if noise; gp.logNoise = hyp[1]; end
    if mean; set_params!(gp.m, hyp[1+noise:noise+num_params(gp.m)]); end
    if kern; set_params!(gp.k, hyp[end-num_params(gp.k)+1:end]); end
end

function show(io::IO, gp::GP)
    println(io, "GP object:")
    println(io, "  Dim = $(gp.dim)")
    println(io, "  Number of observations = $(gp.nobsv)")
    println(io, "  Mean function:")
    show(io, gp.m, 2)
    println(io, "  Kernel:")
    show(io, gp.k, 2)
    println(io, "  Input observations = ")
    show(io, gp.x)
    print(io,"\n  Output observations = ")
    show(io, gp.y)
    print(io,"\n  Variance of observation noise = $(exp(gp.logNoise))")
    print(io,"\n  Marginal Log-Likelihood = ")
    show(io, round(gp.mLL,3))
end
