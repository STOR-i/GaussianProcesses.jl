import StatsBase.predict
import Base.show

# Main GaussianProcess type
type GP
    x::Matrix{Float64}      # Input observations  - each column is an observation
    y::Vector{Float64}      # Output observations
    dim::Int                # Dimension of inputs
    nobsv::Int              # Number of observations
    obsNoise::Float64       # Variance of observation noise
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    # Auxiliary data
    alpha::Vector{Float64}  # (k + obsNoise)⁻¹y
    L::Matrix{Float64}      # Cholesky matrix
    mLL::Float64            # Marginal log-likelihood
    dmLL::Vector{Float64}   # Gradient marginal log-likelihood
    function GP(x::Matrix{Float64}, y::Vector{Float64}, m::Mean, k::Kernel, obsNoise::Float64=0.0)
        dim, nobsv = size(x)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        obsNoise >= 0.0 || throw(ArgumentError("Variance of observation noise must be positive."))
        gp = new(x, y, dim, nobsv, obsNoise, m, k)
        update!(gp)
        return gp
   end
end

# Creates GP object for 1D case
GP(x::Vector{Float64}, y::Vector{Float64}, meanf::Mean, kernel::Kernel, obsNoise::Float64=0.0) = GP(x', y, meanf, kernel, obsNoise)

    
# Update auxiliarly data in GP object after changes have been made
function update!(gp::GP)
    m = meanf(gp.m,gp.x)
    gp.L = chol(crossKern(gp.x,gp.k) + gp.obsNoise*eye(gp.nobsv), :L)
    gp.alpha = gp.L'\(gp.L\(gp.y-m))               
    gp.mLL = -dot((gp.y-m),gp.alpha)/2.0 - sum(log(diag(gp.L))) - gp.nobsv*log(2π)/2.0 #Marginal log-likelihood
    gp.dmLL = Array(Float64, num_params(gp.k))
    Kgrads = grad_stack(gp.x, gp.k)   # [dK/dθᵢ]
    for i in 1:num_params(gp.k)
        #derivative of marginal log-likelihood with respect to kernel hyperparameters 
        gp.dmLL[i] = trace((gp.alpha*gp.alpha' - gp.L'\(gp.L\eye(gp.nobsv)))*Kgrads[:,:,i])/2
    end
           ## Mgrads = #Need a function like grad_stack but for the means
           ##  gp.dmLL.mean[i] = -Mgrads[:,i]'*gp.alpha #Derivative wrt to mean hyperparameters, need to loop over as same with kernel hyperparameters
           ##  gp.dmLL.noise =   gp.obsNoise*trace((gp.alpha*gp.alpha' - gp.L'\(gp.L\eye(gp.nobsv))))  #Derivative wrt the observation noise
end


    
# Given a GP object, predicts the process at requested the points
#
# Arguments:
#  GP Gaussian Process object
#  x  matrix of points for which one would would like to predict the value of the process.
#     (each column of the matrix is a point)
#
# Returns:
# (mu, Sigma) respectively the expected values, lower and upper bounds for values
#             the Gaussian process at the requested locations

function predict(gp::GP, x::Matrix{Float64})
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consisten dimensions"))
    mu = meanf(gp.m,x) + crossKern(x,gp.x,gp.k)*gp.alpha        #Predictive mean 
    Sigma = crossKern(x,gp.k) - ((gp.L\crossKern(x,gp.x,gp.k)')')*(gp.L\crossKern(gp.x,x,gp.k)) #Predictive covariance
    return (mu, Sigma)
end

# 1D Case for prediction
predict(gp::GP, x::Vector{Float64}) = predict(gp, x')

function show(io::IO, gp::GP)
    println(io, "GP object:")
    println(io, " Dim = $(gp.dim)")
    println(io, " Number of observations = $(gp.nobsv)")
    println(io, " Mean function: $(typeof(gp.m))")
    println(io, " Kernel: $(typeof(gp.k))")
    println(io, " Hyperparameters: $(params(gp.k))")
    println(io, " Input observations = ")
    show(io, gp.x)
    print(io,"\n  Output observations = ")
    show(io, gp.y)
    print(io,"\n  Variance of observation noise = $(gp.obsNoise)")
    print(io,"\n  Marginal Log-Likelihood = ")
    show(io, round(gp.mLL,3))
end
