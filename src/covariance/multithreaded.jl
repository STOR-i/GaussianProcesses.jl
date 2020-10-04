function _cov_row!(cK, k, X::AbstractMatrix, data, j, dim)
    cK[j,j] = cov_ij(k, X, X, data, j, j, dim)
    @inbounds for i in 1:j-1
        cK[i,j] = cov_ij(k, X, X, data, i, j, dim)
        cK[j,i] = cK[i,j]
    end
end
function cov!(cK::Matrix, k::Kernel, X::Matrix, data::KernelData=EmptyData())
    dim, nobs = size(X)
    (nobs,nobs) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) and X has size $(size(X))"))
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()] # in case k is not threadsafe (e.g. ADkernel)
    @inbounds Threads.@threads for j in 1:nobs
        kthread = kcopies[Threads.threadid()]
        _cov_row!(cK, k, X, data, j, dim)
    end
    return cK
end
function _cov_row!(cK, k, X1::AbstractMatrix, X2::AbstractMatrix, data, i, dim, nobs2)
    @inbounds for j in 1:nobs2
        cK[i,j] = cov_ij(k, X1, X2, data, i, j, dim)
    end
end
"""
    cov!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())

Like [`cov(k, X1, X2)`](@ref), but stores the result in `cK` rather than a new matrix.
"""
function cov!(cK::Matrix, k::Kernel, X1::Matrix, X2::Matrix, data::KernelData=EmptyData())
    if X1 === X2
        return cov!(cK, k, X1, data)
    end
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    dim = size(X1, 1)
    (nobs1,nobs2) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) X1 $(size(X1)) and X2 $(size(X2))"))
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for i in 1:nobs1
        kthread = kcopies[Threads.threadid()]
        _cov_row!(cK, kthread, X1, X2, data, i, dim, nobs2)
    end
    return cK
end

function _grad_slice_row!(dK, k, X::AbstractMatrix, data, j, p, dim)
    dK[j,j] = dKij_dθp(k,X,X,data,j,j,p,dim)
    @inbounds @simd for i in 1:(j-1)
        dK[i,j] = dKij_dθp(k,X,X,data,i,j,p,dim)
        dK[j,i] = dK[i,j]
    end
end
function grad_slice!(dK::AbstractMatrix, k::Kernel, X::Matrix, data::KernelData, p::Int)
    dim, nobs = size(X)
    (nobs,nobs) == size(dK) || throw(ArgumentError("dK has size $(size(dK)) and X has size $(size(X))"))
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for j in 1:nobs
        kthread = kcopies[Threads.threadid()]
        _grad_slice_row!(dK, kthread, X, data, j, p, dim)
    end
    return dK
end
function _grad_slice_row!(dK, k, X1::AbstractMatrix, X2::AbstractMatrix, data, i, p, dim, nobs2)
    @inbounds @simd for j in 1:nobs2
        dK[i,j] = dKij_dθp(k,X1,X2,data,i,j,p,dim)
    end
end
function grad_slice!(dK::AbstractMatrix, k::Kernel, X1::Matrix, X2::Matrix, data::KernelData, p::Int)
    if X1 === X2
        return grad_slice!(dK, k, X1, data, p)
    end
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    (nobs1,nobs2) == size(dK) || throw(ArgumentError("dK has size $(size(dK)) X1 $(size(X1)) and X2 $(size(X2))"))
    dim=dim1
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for i in 1:nobs1
        kthread = kcopies[Threads.threadid()]
        _grad_slice_row!(dK, kthread, X1, X2, data, i, p, dim, nobs2)
    end
    return dK
end

function _dmll_kern_row!(dmll, buf, k, ααinvcKI, X, data, j, dim, nparams)
    # diagonal
    dKij_dθ!(buf, k, X, X, data, j, j, dim, nparams)
    @inbounds for iparam in 1:nparams
        dmll[iparam] += buf[iparam] * ααinvcKI[j, j] / 2.0
    end
    # off-diagonal
    @inbounds for i in 1:j-1
        dKij_dθ!(buf, k, X, X, data, i, j, dim, nparams)
        @simd for iparam in 1:nparams
            dmll[iparam] += buf[iparam] * ααinvcKI[i, j]
        end
    end
end
"""
    dmll_kern!((dmll::AbstractVector, k::Kernel, X::AbstractMatrix, data::KernelData, ααinvcKI::AbstractMatrix))

Derivative of the marginal log likelihood log p(Y|θ) with respect to the kernel hyperparameters.
"""
function dmll_kern!(dmll::AbstractVector, k::Kernel, X::Matrix, data::KernelData, 
                    ααinvcKI::Matrix{Float64}, covstrat::CovarianceStrategy)
    dim, nobs = size(X)
    nparams = num_params(k)
    @assert nparams == length(dmll)
    dK_buffer = Vector{Float64}(undef, nparams)
    dmll[:] .= 0.0
    # make a copy per thread for objects that are potentially not thread-safe:
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    buffercopies = [similar(dK_buffer) for _ in 1:Threads.nthreads()]
    dmllcopies = [deepcopy(dmll) for _ in 1:Threads.nthreads()]

    @inbounds Threads.@threads for j in 1:nobs
        kthread = kcopies[Threads.threadid()]
        bufthread = buffercopies[Threads.threadid()]
        dmllthread = dmllcopies[Threads.threadid()]
        _dmll_kern_row!(dmllthread, bufthread, kthread, 
                        ααinvcKI, X, data, j, dim, nparams)
    end

    dmll[:] = sum(dmllcopies) # sum up the results from all threads
    return dmll
end
