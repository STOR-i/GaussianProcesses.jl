@doc """
# Description
 Populates D matrix applying a function to each pair of columns of input matrices.
     
# Arguments:
* `D::Matrix{Float64}`  : Output matrix
* `X::Matrix{Float64}`  : Input matrix
* `Y::Matrix{Float64}`  : Input matrix (optional)
* `f::Function`          : Testing function. In this case a function of distance between X and Y
    
# Return:
* `D::Matrix{Float64}`: Matrix D such that `D[i,j] = f(X[:,i], Y[:,j])`
""" ->
function map_column_pairs!{M1<:MatF64,M2<:MatF64,M3<:MatF64}(D::M1, f::Function, X::M2, Y::M3)
    dim, nobs1 = size(X)
    nobs2 = size(Y,2)
    dim == size(Y,1) || throw(ArgumentError("Input observation matrices must have consistent dimensions"))
    size(D,1) == nobs1 || throw(ArgumentError(@sprintf("D has %d rows, while X has %d columns (should be same)", 
                                                       size(D,1), nobs1)))
    size(D,2) == nobs2 || throw(ArgumentError(@sprintf("D has %d columns, while Y has %d columns (should be same)", 
                                                       size(D,2), nobs2)))
    for i in 1:nobs1, j in 1:nobs2
        @inbounds D[i,j] = f(X[:,i], Y[:,j])
    end
    return D
end

@doc """
# Description
 Creates a matrix by applying a function to each pair of columns of input matrices.
     
# Arguments:
* `X::Matrix{Float64}`  : Input matrix
* `Y::Matrix{Float64}`  : Input matrix (optional)
* `f::Function`          : Testing function. In this case a function of distance between X and Y
    
# Return:
* `D::Matrix{Float64}`: Symmetric matrix D such that `D[i,j] = f(X[:,i], Y[:,j])`
""" ->
function map_column_pairs{M1<:MatF64,M2<:MatF64}(f::Function, X::M1, Y::M2)
    nobs1 = size(X,2)
    nobs2 = size(Y,2)
    D= Array(Float64, nobs1, nobs2)
    map_column_pairs!(D, f, X, Y)
    return D
end

@doc """
# Description
Populates D matrix by applying a function to each pair of columns of an input matrix.

# Arguments
* `D::AbstractMatrix{Float64}`  : Output matrix
* `X::Matrix{Float64}`: matrix of observations (each column is an observation)
* `d::Function`: function specifying covariance between two points

# Return:
* `D::Matrix{Float64}`: Symmetric matrix D such that `D[i,j] = d(X[:,i], X[:,j])`

""" ->
function map_column_pairs!{M1<:MatF64,M2<:MatF64}(D::M1, f::Function, X::M2)
    dim, nobsv = size(X)
    size(D,1) == nobsv || throw(ArgumentError(@sprintf("D has %d rows, while X has %d columns (should be same)", 
                                                       size(D,1), nobsv)))
    size(D,2) == nobsv || throw(ArgumentError(@sprintf("D has %d columns, while X has %d columns (should be same)", 
                                                       size(D,2), nobsv)))
    for i in 1:nobsv
        for j in 1:i
            @inbounds D[i,j] = f(X[:,i], X[:,j])
            if i != j; @inbounds D[j,i] = D[i,j]; end;
        end
    end
    return D
end

@doc """
# Description
Constructs matrix by applying a function to each pair of columns of an input matrix.

# Arguments
* `X::Matrix{Float64}`: matrix of observations (each column is an observation)
* `d::Function`: function specifying covariance between two points
# Return:
* `D::Matrix{Float64}`: Symmetric matrix D such that `D[i,j] = d(X[:,i], X[:,j])`

""" ->
function map_column_pairs{M<:MatF64}(f::Function, X::M)
    dim, nobsv = size(X)
    D = Array(Float64, nobsv, nobsv)
    for i in 1:nobsv
        for j in 1:i
            @inbounds D[i,j] = f(X[:,i], X[:,j])
            if i != j; @inbounds D[j,i] = D[i,j]; end;
        end
    end
    return D
end


# Taken from Distributions package
φ(z::Real) = exp(-0.5*z*z)/√2π
Φ(z::Real) = 0.5*erfc(-z/√2)
