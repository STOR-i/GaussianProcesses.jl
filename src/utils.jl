@doc """
# Description
 Creates a kernel matrix from vector inputs according to a function d, where D[i,j] = d(x1[i], x2[j]) or D[i,j] = d(x1[i], x1[j]). For example, where d is function of distance between x1 and x2.
# Arguments:
* `x1::Matrix{Float64}`  : Input matrix
* `x2::Matrix{Float64}`  : Input matrix (optional)
* `d::Function`          : Testing function. In this case a function of distance between x1 and x2
# Returns:
* `D::Matrix{Float64}`: A positive definite matrix given as output from d(x1,x2)
""" ->
function crossKern(x1::Matrix{Float64}, x2::Matrix{Float64}, d::Function)
    dim, nobs1 = size(x1)
    nobs2 = size(x2,2)
    dim == size(x2,1) || throw(ArgumentError("Input observation matrices must have consistent dimensions"))
    D= Array(Float64, nobs1, nobs2)
    for i in 1:nobs1, j in 1:nobs2
        @inbounds D[i,j] = d(x1[:,i], x2[:,j])
    end
    return max(D,0)
end

# Returns matrix D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
#  d is a function between two vectors
function crossKern(x::Matrix{Float64}, d::Function)
    dim, nobsv = size(x)
    D = Array(Float64, nobsv, nobsv)
    for i in 1:nobsv
        for j in 1:i
            @inbounds D[i,j] = d(x[:,i], x[:,j])
            if i != j; @inbounds D[j,i] = D[i,j]; end;
        end
    end
    return max(D,0)
end

# Calculates the stack [dm / dθᵢ] of mean matrix gradients
function grad_stack(x::Matrix{Float64}, m::Mean)
    n = num_params(m)
    d, nobsv = size(x)
    mat = Array(Float64, nobsv, n)
    for i in 1:nobsv
        @inbounds mat[i,:] = grad_meanf(m, x[:,i])
    end
    return mat
end

# Taken from Distributions package
φ(z::Real) = exp(-0.5*z*z)/√2π
Φ(z::Real) = 0.5*erfc(-z/√2)
