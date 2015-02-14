import StatsBase.predict

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

type GaussianProcess
    x::Matrix{Float64}   # Input observations  - each column is an observation
    y::Vector{Float64}   # Output observations
    dim::Int             # Dimension of inputs
    nobvs::Int           # Number of observations
    kernel::Function     # Function which takes two vectors as argument and returns the distance between them
    _cov_xx_inv::Matrix{Float64} 
    function GaussianProcess(x::Matrix{Float64}, y::Vector{Float64}, kernel::Function)
        dim, nobvs = size(x)
        length(y) == nobvs || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        _cov_xx_inv = inv(distance(x,kernel))
        new(x, y, dim, nobvs, kernel, _cov_xx_inv)
    end
end

# Creates GaussianProcess object for 1D case
GaussianProcess(x::Vector{Float64}, y::Vector{Float64}, kernel::Function) = GaussianProcess(x', y, kernel)

# Given a GaussianProcess object, predicts
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
function predict(GP::GaussianProcess, x::Matrix{Float64})
    size(x,1) == GP.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consisten dimensions"))
    exp = distance(x, GP.x, GP.kernel) * GP._cov_xx_inv * GP.y
    cf = distance(x,GP.kernel) - distance(x,GP.x,GP.kernel) *GP._cov_xx_inv * distance(GP.x, x, GP.kernel)  # Covariance function
    conf = 2*sqrt(max(diag(cf), 0.0))
    u = exp + conf
    l = exp - conf
    return (exp, l, u)
end

# 1D Case for prediction
predict(GP::GaussianProcess, x::Vector{Float64}) = predict(GP, x')
