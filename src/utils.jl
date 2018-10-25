"""
    map_column_pairs(f, X::Matrix{Float64}[, Y::Matrix{Float64} = X])


Create a matrix by applying function `f` to each pair of columns of input matrices
`X` and `Y`.
"""
function map_column_pairs(f, X::MatF64, Y::AbstractArray{T, 2}) where T
    nobs1 = size(X,2)
    nobs2 = size(Y,2)
    D = Array{T}(undef, nobs1, nobs2)
    map_column_pairs!(D, f, X, Y)
    return D
end

function map_column_pairs(f, X::AbstractArray{T, 2}) where T
    dim, nobsv = size(X)
    D = Array{T}(undef, nobsv, nobsv)
    for i in 1:nobsv
        for j in 1:i
            @inbounds D[i,j] = f(X[:,i], X[:,j])
            if i != j; @inbounds D[j,i] = D[i,j]; end;
        end
    end
    return D
end

"""
    map_column_pairs!(D::Matrix{Float64}, f, X::Matrix{Float64}[, Y::Matrix{Float64} = X])

Like [`map_column_pairs`](@ref), but stores the result in `D` rather than a new matrix.
"""
function map_column_pairs!(D::MatF64, f, X::MatF64, Y::MatF64)
    dim, nobs1 = size(X)
    nobs2 = size(Y,2)
    dim == size(Y,1) || throw(ArgumentError("Input observation matrices must have consistent dimensions"))
    size(D,1) == nobs1 || throw(ArgumentError(@sprintf("D has %d rows, while X has %d columns (should be same)",
                                                       size(D,1), nobs1)))
    size(D,2) == nobs2 || throw(ArgumentError(@sprintf("D has %d columns, while Y has %d columns (should be same)",
                                                       size(D,2), nobs2)))
    for i in 1:nobs1, j in 1:nobs2
        @inbounds D[i,j] = f(view(X,:,i), view(Y,:,j))
    end
    return D
end

function map_column_pairs!(D::MatF64, f, X::MatF64)
    dim, nobsv = size(X)
    size(D,1) == nobsv || throw(ArgumentError(@sprintf("D has %d rows, while X has %d columns (should be same)",
                                                       size(D,1), nobsv)))
    size(D,2) == nobsv || throw(ArgumentError(@sprintf("D has %d columns, while X has %d columns (should be same)",
                                                       size(D,2), nobsv)))
    @inbounds for i in 1:nobsv
        for j in 1:i
            D[i,j] = f(view(X,:,i), view(X,:,j))
            if i != j; D[j,i] = D[i,j]; end;
        end
    end
    return D
end
