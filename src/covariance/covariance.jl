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

"""
    cov(k::Kernel, X::AbstractMatrix[, data::KernelData = EmptyData()])

Create covariance matrix from kernel `k`, matrix of observations `X`, where each column is
an observation, and kernel data `data` constructed from input observations.
"""
cov(k::Kernel, X::AbstractMatrix, data::KernelData=EmptyData()) = cov(k, X, X, data)

function cov!(cK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData=EmptyData())
    dim, nobs1 = size(X1)
    dim, nobs2 = size(X2)
    cK .= cov_ij.(Ref(k), Ref(X1), Ref(X2), Ref(data), 1:nobs1, (1:nobs2)', dim)
end
cov!(cK::AbstractMatrix, k::Kernel, X::AbstractMatrix, data::KernelData=EmptyData()) = cov!(cK, k, X, X, data)

@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = cov(k, @view(X1[:,i]), @view(X2[:,j]))
# the default is to drop the KernelData
@inline cov_ij(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, dim::Int) = cov_ij(k, X1, X2, i, j, dim)

############################
##### Kernel Gradients #####
############################
@inline @inbounds function dKij_dθ!(dK::AbstractVector, kern::Kernel, X1::AbstractMatrix, X2::AbstractMatrix,
                                    data::KernelData, i::Int, j::Int, dim::Int, npars::Int)
    for iparam in 1:npars
        dK[iparam] = dKij_dθp(kern, X1, X2, data, i, j, iparam, dim)
    end
end

# Calculates the stack [dk / dθᵢ] of kernel matrix gradients
function grad_stack!(stack::AbstractArray, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData)
    @inbounds for p in 1:num_params(k)
        grad_slice!(view(stack, :, :, p), k, X1, X2, data, p)
    end
    stack
end

grad_stack!(stack::AbstractArray, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) =
    grad_stack!(stack, k, X1, X2, KernelData(k, X1, X2))

grad_stack(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix) = grad_stack(k, X1, X2, KernelData(k, X1, X2))

function grad_stack(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData)
    nobs1 = size(X1, 2)
    nobs2 = size(X2, 2)
    stack = Array{eltype(X)}(undef, nobs1, nobs2, num_params(k))
    grad_stack!(stack, k, X1, X2, data)
end

function grad_slice!(dK::AbstractMatrix, k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, p::Int)
    dim, nobs1 = size(X1)
    dim, nobs2 = size(X2)
    dK .= dKij_dθp.(Ref(k), Ref(X1), Ref(X2), Ref(data), 1:nobs1, (1:nobs2)', p, dim)
end

@inline function dKij_dθp(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix, data::KernelData, i::Int, j::Int, p::Int, dim::Int) 
    return dKij_dθp(k, X1, X2, i, j, p, dim)
end
