# This file contains the code for differentiating the Cholesky decomposition

function level2partition(A::Matrix{Float64}, j::Int)
    N = size(A, 1)
    r = view(A, j, 1:j-1)
    d = view(A, j, j)
    B = view(A, j+1:N , 1:j-1)
    c = view(A, j+1:N, j)
    return r, d, B, c
end

function chol_unblocked_rev!(L::Matrix{Float64}, A_bar)
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
        
        LinAlg.BLAS.gemv!('T', -1.0, B, c_bar, 1.0, r_bar) # r_bar = r_bar - Bᵀ c_bar
        LinAlg.BLAS.ger!(-1.0, c_bar, r, B_bar) # B_bar = B_bar - c_bar * r'
        d_bar[1] /= 2
    end
    
    return A_bar
end
