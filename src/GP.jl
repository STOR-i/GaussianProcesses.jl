import Base.show
import ScikitLearnBase

# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.

# Constructors:
    GP(x, y, m, k, logNoise)
    GP(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8) # ScikitLearn constructor

# Arguments:
* `x::Matrix{Float64}`: Training inputs
* `y::Vector{Float64}`: Observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `logNoise::Float64` : Log of the observation noise. The default is -1e8, which is equivalent to assuming no observation noise.

# Returns:
* `gp::GP`            : A Gaussian process fitted to the training data
""" ->
type GP
    x::Matrix{Float64}      # Input observations  - each column is an observation
    y::Vector{Float64}      # Output observations
    dim::Int                # Dimension of inputs
    nobsv::Int              # Number of observations
    logNoise::Float64       # log variance of observation noise
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    # Auxiliary data
    cK::AbstractPDMat       # (k + obsNoise)
    alpha::Vector{Float64}  # (k + obsNoise)⁻¹y
    mLL::Float64            # Marginal log-likelihood
    dmLL::Vector{Float64}   # Gradient marginal log-likelihood
    function GP(x::Matrix{Float64}, y::Vector{Float64}, m::Mean, k::Kernel, logNoise::Float64=-1e8)
        dim, nobsv = size(x)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(x, y, dim, nobsv, logNoise, m, k)
        update_mll!(gp)
        return gp
    end
    # ScikitLearn.jl constructor
    function GP(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8)
        # We could leave x/y/dim/nobsv undefined if we reordered the fields
        new(zeros(Float64,0,0), zeros(Float64, 0), 0, 0, logNoise, m, k)
    end
end

# Creates GP object for 1D case
GP(x::Vector{Float64}, y::Vector{Float64}, meanf::Mean, kernel::Kernel, logNoise::Float64=-1e8) = GP(x', y, meanf, kernel, logNoise)

ScikitLearnBase.declare_hyperparameters(GP, [:m, :k, :logNoise])
ScikitLearnBase.is_classifier(::GP) = false

function ScikitLearnBase.fit!(gp::GP, X::Matrix{Float64}, y::Vector{Float64})
    gp.x = X' # ScikitLearn's X is (n_samples, n_features)
    gp.y = y
    gp.dim, gp.nobsv = size(gp.x)
    length(y) == gp.nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    update_mll!(gp)
    return gp
end

# Update auxiliarly data in GP object after changes have been made
function update_mll!(gp::GP)
    m = meanf(gp.m,gp.x)
    gp.cK = PDMat(crossKern(gp.x,gp.k) + exp(gp.logNoise)*eye(gp.nobsv))
    gp.alpha = gp.cK \ (gp.y - m)
    gp.mLL = -dot((gp.y-m),gp.alpha)/2.0 - logdet(gp.cK)/2.0 - gp.nobsv*log(2π)/2.0 #Marginal log-likelihood
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

# Returns:
* `(mu, Sigma)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
                                                    process at the specified points
""" ->
function predict(gp::GP, x::Matrix{Float64}; full_cov::Bool=false)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    if full_cov
        return _predict(gp, x)
    else
        ## calculate prediction for each point independently
            mu = Array(Float64, size(x,2))
            Sigma = similar(mu)
        for k in 1:size(x,2)
            out = _predict(gp, x[:,k:k])
            mu[k] = out[1][1]
            Sigma[k] = max(out[2][1],0)
        end
        return mu, Sigma
    end
end

function ScikitLearnBase.predict(gp::GP, X::Matrix{Float64}; eval_MSE::Bool=false)
    mu, Sigma = predict(gp, X'; full_cov=false)
    if eval_MSE
        return mu, Sigma
    else
        return mu
    end
end

# 1D Case for prediction
predict(gp::GP, x::Vector{Float64};full_cov::Bool=false) = predict(gp, x'; full_cov=full_cov)

## compute predictions
function _predict(gp::GP, x::Array{Float64})
    cK = crossKern(x,gp.x,gp.k)
    Lck = whiten(gp.cK, cK')
    mu = meanf(gp.m,x) + cK*gp.alpha    # Predictive mean
    Sigma = crossKern(x,gp.k) - Lck'Lck # Predictive covariance
    return (mu, Sigma)
end


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
