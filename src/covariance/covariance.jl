"""
    cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix)

Create covariance matrix from kernel `k` and matrices of observations `X1` and `X2`, where
each column is an observation.
"""
function cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    cK = Array{promote_type(eltype(X1), eltype(X2))}(undef, nobs1, nobs2)
    cov!(cK, k, X1, X2, data)
end

function _cov_row!(cK, k, X::AbstractMatrix, data, j, dim)
    cK[j,j] = cov_ij(k, X, X, data, j, j, dim)
    @inbounds for i in 1:j-1
        cK[i,j] = cov_ij(k, X, X, data, i, j, dim)
        cK[j,i] = cK[i,j]
    end
end
function cov_loop!(cK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::KernelData)
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
function cov_loop!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData)
    if X1 === X2
        return cov_loop!(cK, k, X1, data)
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
"""
    cov(k::Kernel, X::AbstractMatrix[, data::KernelData = EmptyData()])

Create covariance matrix from kernel `k`, matrix of observations `X`, where each column is
an observation, and kernel data `data` constructed from input observations.
"""
cov(k::Kernel, X::AbstractMatrix, data::KernelData=EmptyData()) = cov(k, X, X, data)

function cov_loop_generic!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData)
    dim, nobs1 = size(X1)
    dim, nobs2 = size(X2)
    # Generic implementation using broadcasting. A more efficient multithreaded
    # implementation is provided above.
    cK .= cov_ij.(Ref(k), Ref(X1), Ref(X2), Ref(data), 1:nobs1, (1:nobs2)', dim) .* 2.0
end
"""
    cov!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())

Like [`cov(k, X1, X2)`](@ref), but stores the result in `cK` rather than a new matrix.
"""
cov!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix=X1, data::KernelData=EmptyData()) = cov_loop!(cK, k, X1, X2, data)
cov!(cK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::KernelData=EmptyData()) = cov!(cK, k, X, X, data)

############################
##### Kernel Gradients #####
############################

function _grad_slice_row!(dK, k, X::AbstractMatrix, data, j, p, dim)
    dK[j,j] = dKij_dθp(k,X,X,data,j,j,p,dim)
    @inbounds @simd for i in 1:(j-1)
        dK[i,j] = dKij_dθp(k,X,X,data,i,j,p,dim)
        dK[j,i] = dK[i,j]
    end
end
function grad_slice_loop!(dK::AbstractMatrix, k::Kernel, X::AbstractMatrix, p::Int, data::KernelData)
    dim, nobs = size(X)
    (nobs,nobs) == size(dK) || throw(ArgumentError("dK has size $(size(dK)) and X has size $(size(X))"))
    kcopies = [deepcopy(k) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for j in 1:nobs
        kthread = kcopies[Threads.threadid()]
        _grad_slice_row!(dK, kthread, X, data, j, p, dim)
    end
    return dK
end
function _grad_slice_row!(dK, k, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i, p, dim, nobs2)
    @inbounds @simd for j in 1:nobs2
        dK[i,j] = dKij_dθp(k,X1,X2,data,i,j,p,dim)
    end
end
function grad_slice_loop!(dK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, p::Int, data::KernelData)
    if X1 === X2
        return grad_slice_loop!(dK, k, X1, p, data)
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
function grad_slice_loop_generic!(dK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, p::Int, data::KernelData)
    dim, nobs1 = size(X1)
    dim, nobs2 = size(X2)
    # Generic implementation using broadcasting. A more efficient multithreaded
    # implementation is provided above.
    dK .= dKij_dθp.(Ref(k), Ref(X1), Ref(X2), Ref(data), 1:nobs1, (1:nobs2)', p, dim)
end
grad_slice!(dK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, p::Int, data::KernelData=EmptyData()) = grad_slice_loop!(dK, k, X1, X2, p, data)

# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!(stack::AbstractArray, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())
    @inbounds for p in 1:num_params(k)
        grad_slice!(view(stack, :, :, p), k, X1, X2, p, data)
    end
    stack
end

# grad_stack!(stack::AbstractArray, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) =
    # grad_stack!(stack, k, X1, X2, KernelData(k, X1, X2))

# grad_stack(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) = grad_stack(k, X1, X2, KernelData(k, X1, X2))

function grad_stack(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())
    nobs1 = size(X1, 2)
    nobs2 = size(X2, 2)
    stack = Array{eltype(X)}(undef, nobs1, nobs2, num_params(k))
    grad_stack!(stack, k, X1, X2, data)
end


@inline function dKij_dθp(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int) 
    return dKij_dθp(k, X1, X2, i, j, p, dim)
end
