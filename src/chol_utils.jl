# This file contains the code for differentiating the Cholesky decomposition

function level2partition{T<:Real}(A::DenseMatrix{T}, j::Int)
    N = size(A, 1)
    r = view(A, j, 1:j-1)
    d = view(A, j, j)
    B = view(A, j+1:N , 1:j-1)
    c = view(A, j+1:N, j)
    return r, d, B, c
end

# Derivative of Cholesky decomposition, see Murray(2016). Differentiation of the Cholesky decomposition. arXiv.1602.07527
function chol_unblocked_rev!{T<:Real}(L::DenseMatrix{T}, A_bar::DenseMatrix{T})
    N = size(A_bar, 1)
    for j in N:-1:1
        r, d, B, c = level2partition(L, j)
        r_bar, d_bar, B_bar, c_bar = level2partition(A_bar, j)
        d_bar[1] -= vecdot(c_bar, c)/d[1]
        d_bar[1] /= d[1]

        for i in eachindex(c_bar)
            c_bar[i] /= d[1]
        end
        
        for i in eachindex(r_bar)
            r_bar[i] -= d_bar[1]*r[i]
        end

        r_bar[:] = r_bar - B'c_bar
        # LinAlg.BLAS.gemv!('T', -1.0, B, c_bar, 1.0, r_bar) # r_bar = r_bar - Bᵀ c_bar
        B_bar[:,:] = B_bar - c_bar * r'
        # LinAlg.BLAS.ger!(-1.0, c_bar, r, B_bar) # B_bar = B_bar - c_bar * r'
        d_bar[1] /= 2
    end
    
    return A_bar
end

