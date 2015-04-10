import Base.show

# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.
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
    alpha::Vector{Float64}  # (k + obsNoise)⁻¹y
    L::Matrix{Float64}      # Cholesky matrix
    mLL::Float64            # Marginal log-likelihood
    dmLL::Vector{Float64}   # Gradient marginal log-likelihood
    function GP(x::Matrix{Float64}, y::Vector{Float64}, m::Mean, k::Kernel, logNoise::Float64=-1e8)
        dim, nobsv = size(x)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(x, y, dim, nobsv, logNoise, m, k)
        update!(gp)
        return gp
   end
end

# Creates GP object for 1D case
GP(x::Vector{Float64}, y::Vector{Float64}, meanf::Mean, kernel::Kernel, logNoise::Float64=-1e8) = GP(x', y, meanf, kernel, logNoise)

# Update auxiliarly data in GP object after changes have been made
function update!(gp::GP)
    m = meanf(gp.m,gp.x)
    gp.L = chol(crossKern(gp.x,gp.k) + exp(gp.logNoise)*eye(gp.nobsv), :L)
    gp.alpha = gp.L'\(gp.L\(gp.y-m))               
    gp.mLL = -dot((gp.y-m),gp.alpha)/2.0 - sum(log(diag(gp.L))) - gp.nobsv*log(2π)/2.0 #Marginal log-likelihood
    gp.dmLL = Array(Float64, 1+ num_params(gp.m) + num_params(gp.k))

    # Calculate Gradient with respect to hyperparameters
    gp.dmLL[1] = exp(2*gp.logNoise)*trace((gp.alpha*gp.alpha' - gp.L'\(gp.L\eye(gp.nobsv))))  #Derivative wrt the observation noise

    Mgrads = grad_stack(gp.x, gp.m)
    for i in 1:num_params(gp.m)
        gp.dmLL[i+1] = -dot(Mgrads[:,i],gp.alpha) #Derivative wrt to mean hyperparameters, need to loop over as same with kernel hyperparameters
    end
    
    # Derivative of marginal log-likelihood with respect to kernel hyperparameters
    Kgrads = grad_stack(gp.x, gp.k)   # [dK/dθᵢ]
    for i in 1:num_params(gp.k)
        gp.dmLL[i+num_params(gp.m)+1] = trace((gp.alpha*gp.alpha' - gp.L'\(gp.L\eye(gp.nobsv)))*Kgrads[:,:,i])/2
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
function predict(gp::GP, x::Matrix{Float64})
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    mu = meanf(gp.m,x) + crossKern(x,gp.x,gp.k)*gp.alpha        #Predictive mean 
    Sigma = crossKern(x,gp.k) - ((gp.L\crossKern(x,gp.x,gp.k)')')*(gp.L\crossKern(gp.x,x,gp.k)) #Predictive covariance
    return (mu, Sigma)
end

# 1D Case for prediction
predict(gp::GP, x::Vector{Float64}) = predict(gp, x')


get_params(gp::GP) = [gp.logNoise, get_params(gp.m), get_params(gp.k)]
function set_params!(gp::GP, hyp::Vector{Float64})
    gp.logNoise = hyp[1]
    set_params!(gp.m, hyp[2:1+num_params(gp.m)])
    set_params!(gp.k, hyp[end-num_params(gp.k)+1:end])
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
