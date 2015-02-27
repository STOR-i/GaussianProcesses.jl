import StatsBase.predict
import Base.show

# Returns matrix of distances D where D[i,j] = kernel(x1[i], x2[j])
#
# Arguments:
#  x1 matrix of observations (each column is an observation)
#  x2 matrix of observations (each column is an observation)
function distance(x1::Matrix{Float64}, x2::Matrix{Float64}, k::Kernel)
    dim, nobs1 = size(x1)
    nobs2 = size(x2,2)
    dim == size(x2,1) || throw(ArgumentError("Input observation matrices must have consistent dimensions"))
    dist = Array(Float64, nobs1, nobs2)
    for i in 1:nobs1, j in 1:nobs2
        dist[i,j] = kern(k, x1[:,i], x2[:,j])
    end
    return dist
end

# Returns PD matrix of distances D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
function distance(x::Matrix{Float64}, k::Kernel)
    dim, nobsv = size(x)
    dist = Array(Float64, nobsv, nobsv)
    for i in 1:nobsv
        for j in 1:i
            dist[i,j] = kern(k, x[:,i], x[:,j])
            if i != j; dist[j,i] = dist[i,j]; end;
        end
    end
    return dist
end

# Main GaussianProcess type
type GP
    x::Matrix{Float64}   # Input observations  - each column is an observation
    y::Vector{Float64}   # Output observations
    dim::Int             # Dimension of inputs
    nobsv::Int           # Number of observations
    obsNoise::Float64    # Variance of observation noise
    meanf::Function      # Mean function
    k::Kernel            # Kernel object
    alpha::Matrix{Float64} 
    L::Matrix{Float64}  # Cholesky martrix
    mLL::Float64        # Marginal Log-likelihood
    
    function GP(x::Matrix{Float64}, y::Vector{Float64}, meanf::Function, k::Kernel, obsNoise::Float64=0.0)
        dim, nobsv = size(x)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        m = meanf(x)
        L = chol(distance(x,k)+obsNoise*eye(nobsv))'     #Cholesky factorisation
        alpha = (L'\(L\(y-m)))
        mLL = -(y-m)'*alpha/2 - sum(log(diag(L))) - nobsv*log(2*pi)/2   #marginal log-likelihood
     #   dmLL = trace((alpha*alpha' - L'\(L\eye(2)))*grad_kern(k,distance(x,k)))/2 #derivative of marginal log-likelihood with respect to hyperparameters
        new(x, y, dim, nobsv, obsNoise, meanf, k, alpha, L, mLL)
   end
end

# Creates GP object for 1D case
GP(x::Vector{Float64}, y::Vector{Float64}, meanf::Function, kernel::Kernel, obsNoise::Float64=0.0) = GP(x', y, meanf, kernel, obsNoise)

# Given a GP object, predicts
# with confidence bounds the value of the process
# requested points
#
# Arguments:
#  GP Gaussian Process object
#  x  matrix of points for which one would would like to predict the value of the process.
#     (each column of the matrix is a point)
#
# Returns:
# (exp, l, u) respectively the expected values, lower and upper bounds for values
#             the Gaussian process at the requested locations

function predict(gp::GP, x::Matrix{Float64})
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consisten dimensions"))
    mu = gp.meanf(x) + distance(x,gp.x,gp.k)*gp.alpha        #Predictive mean 
    Sigma = distance(x,gp.k) - ((gp.L\distance(x,gp.x,gp.k)')')*(gp.L\distance(gp.x,x,gp.k)) #Predictive covariance
    return (mu, Sigma)
end

# 1D Case for prediction
predict(gp::GP, x::Vector{Float64}) = predict(gp, x')

function show(io::IO, gp::GP)
    println(io, "GP object:")
    println(io, " Dim = $(gp.dim)")
    println(io, " Number of observations = $(gp.nobsv)")
    println(io, " Hyperparameters: $(params(gp.k))")
    println(io, " Input observations = ")
    show(io, gp.x)
    print(io,"\n  Output observations = ")
    show(io, gp.y)
    print(io,"\n  Variance of observation noise = $(gp.obsNoise)")
    print(io,"\n  Marginal Log-Likelihood = ")
    show(io, round(gp.mLL,3))
end




