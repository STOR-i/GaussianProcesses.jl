import StatsBase.predict
import Base.show

# Returns matrix of distances D where D[i,j] = kernel(x1[i], x2[j])
#
# Arguments:
#  x1 matrix of observations (each column is an observation)
#  x2 matrix of observations (each column is an observation)
function distance(x1::Matrix{Float64}, x2::Matrix{Float64}, kernel::Function)
    dim, nobs1 = size(x1)
    nobs2 = size(x2,2)
    dim == size(x2,1) || throw(ArgumentError("Input observation matrices must have consistent dimensions"))
    dist = Array(Float64, nobs1, nobs2)
    for i in 1:nobs1, j in 1:nobs2
        dist[i,j] = kernel(x1[:,i], x2[:,j])
    end
    return dist
end

# Returns PD matrix of distances D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
function distance(x::Matrix{Float64}, kernel::Function)
    dim, nobsv = size(x)
    dist = Array(Float64, nobsv, nobsv)
    for i in 1:nobsv
        for j in 1:i
            dist[i,j] = kernel(x[:,i], x[:,j])
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
    meanf::Function      # Mean function
    kernel::Function     # Function which takes two vectors as argument and returns the distance between them
    alpha::Matrix{Float64} 
    L::Matrix{Float64}  # Cholesky martrix
    mLL::Float64        # Marginal Log-likelihood
    
    function GP(x::Matrix{Float64}, y::Vector{Float64}, meanf::Function, kernel::Function)
        dim, nobsv = size(x)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))

        m = meanf(x)
        L = chol(distance(x,kernel))'     #Cholesky factorisation
        alpha = (L'\(L\(y-m)))'
        mLL = dot((y-m),alpha)/2 + sum(log(diag(L))) + nobsv*log(2*pi)/2   #marginal log-likelihood
        new(x, y, dim, nobsv, meanf, kernel, alpha, L, mLL)
    end
end

# Creates GP object for 1D case
GP(x::Vector{Float64}, y::Vector{Float64}, meanf::Function, kernel::Function) = GP(x', y, meanf, kernel)

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
    mu = gp.meanf(x) + distance(x,gp.x,gp.kernel)*gp.alpha'
    Sigma = distance(x,gp.kernel) - ((gp.L\distance(x,gp.x,gp.kernel)')')*(gp.L\distance(gp.x,x,gp.kernel))
    return (mu, Sigma)
end

# 1D Case for prediction
predict(gp::GP, x::Vector{Float64}) = predict(gp, x')

function show(io::IO, gp::GP)
    println(io, "GP object:")
    println(io, "  Dim = $(gp.dim)")
    println(io, "  Number of observations = $(gp.nobsv)")
    println(io, "  Input observations = ")
    show(io, gp.x)
    print(io,"\n  Output observations = ")
    show(io, gp.y)
    print(io,"\n  Marginal Log-Likelihood = ")
    show(io, round(gp.mLL,3))
end


