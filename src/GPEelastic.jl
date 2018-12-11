import Base.append!
append!(gp, x::AbstractVector, y::Float64) = append!(gp, reshape(x, :, 1), [y])
function append!(gp::GPE{X,Y,M,K,P,D}, x::AbstractMatrix, y::AbstractVector) where {X,Y,M,K,P <: ElasticPDMat, D}
    size(x, 2) == length(y) || error("$(size(x, 2)) observations, but $(length(y)) targets.")
    newcov = [cov(gp.kernel, gp.x, x); cov(gp.kernel, x, x) + (exp(2*gp.logNoise) + eps())*I]
    append!(gp.data, gp.kernel, gp.x, x)
    append!(gp.x, x)
    append!(gp.cK, newcov)
    gp.nobs += length(y)
    append!(gp.y, y)
    update_target!(gp, kern = false, noise = false)
end

wrap_cK(cK::ElasticPDMat, Σbuffer, chol::Cholesky) = cK
mat(cK::ElasticPDMat) = view(cK.mat)
cholfactors(cK::ElasticPDMat) = view(cK.chol).factors

function ElasticGPE(dim; mean::Mean = MeanZero(), kernel = SE(0.0, 0.0),
                    logNoise::Float64 = -2.0, kwargs...)
    x = ElasticArray(Array{Float64}(undef, dim, 0))
    y = ElasticArray(Array{Float64}(undef, 0))
    ElasticGPE(x, y, mean, kernel, logNoise; kwargs...)
end
function ElasticGPE(x::AbstractMatrix, y::AbstractVector, mean::Mean, kernel::Kernel, 
                    logNoise::Float64 = -2.0;
                    capacity = 10^3, stepsize = 10^3)
    data = ElasticKernelData(kernel, x, x, capacity = capacity, stepsize = stepsize)
    nobs = length(y)
    # create placeholder PDMat
    m    = Matrix{Float64}(undef, nobs, nobs)
    chol = Matrix{Float64}(undef, nobs, nobs)
    cK = ElasticPDMat(m, Cholesky(chol, :U, 0), capacity=capacity, stepsize=stepsize)
    gp = GPE(ElasticArray(x), ElasticArray(y), mean, kernel, logNoise, data, cK)
    initialise_target!(gp)
end

function prepareappend!(kd, Xnew)
    dim, nobs_new = size(Xnew)
    nobs = kd.dims[1]
    if nobs + nobs_new > kd.capacity[1] 
        kd.capacity = kd.capacity .+ kd.stepsize
        ElasticPDMats.resize!(kd)
    end
    kd, dim, nobs, nobs_new
end

function ElasticKernelData(k::Isotropic, X1::AbstractMatrix, X2::AbstractMatrix; capacity = 10^3, stepsize = 10^3)
    @assert X1==X2 # TODO: extend to X1 != X2
    X = X1
    kerneldata = IsotropicData(AllElasticArray(2; capacity = (capacity, capacity), stepsize = (stepsize, stepsize)))
    nobs = size(X, 2)
    distance!(view(kerneldata.R, 1:nobs, 1:nobs), k, X)
    setdimension!(kerneldata.R, nobs, 1:2)
    kerneldata
end
function append!(kerneldata::IsotropicData{<:AllElasticArray}, k::Isotropic, X::AbstractMatrix, Xnew::AbstractMatrix)
    kd, dim, nobs, nobs_new = prepareappend!(kerneldata.R, Xnew)
    distance!(view(kd, 1:nobs, nobs + 1:nobs + nobs_new), k, X, Xnew)
    copyto!(view(kd, nobs + 1:nobs + nobs_new, 1:nobs), transpose(view(kd, 1:nobs, nobs + 1:nobs + nobs_new))) 
    distance!(view(kd, nobs + 1:nobs + nobs_new, nobs + 1:nobs + nobs_new), k, Xnew)
    setdimension!(kd, nobs + nobs_new, 1:2)
    kerneldata
end
append!(kerneldata, k, X, Xnew::AbstractVector) = append!(kerneldata, k, X, reshape(Xnew, :, 1))

function ElasticKernelData(k::StationaryARD, X1::AbstractMatrix, X2::AbstractMatrix; capacity = 10^3, stepsize = 10^3)
    @assert X1==X2 # TODO: extend to X1 != X2
    X = X1
    dim, nobs = size(X)
    dist_stack = AllElasticArray(3; capacity = (capacity, capacity, size(X, 1)),
                                    stepsize = (stepsize, stepsize, 0))
    for d in 1:dim
        grad_ls = view(dist_stack, 1:nobs, 1:nobs, d)
        distance!(grad_ls, SqEuclidean(), view(X, d:d, :))
    end
    setdimension!(dist_stack, nobs, 1:2)
    setdimension!(dist_stack, dim, 3)
    StationaryARDData(dist_stack)
end
function append!(kerneldata::StationaryARDData, kernel::StationaryARD, X::AbstractMatrix, Xnew::AbstractMatrix)
    kd, dim, nobs, nobs_new = prepareappend!(kerneldata.dist_stack, Xnew)
    for d in 1:dim
        grad_ls = view(kd, 1:nobs, nobs + 1:nobs + nobs_new, d)
        distance!(grad_ls, SqEuclidean(), view(X, d:d, :), view(Xnew, d:d, :))
        copyto!(view(kd, nobs + 1:nobs + nobs_new, 1:nobs, d),
                transpose(grad_ls))
        distance!(view(kd, nobs + 1:nobs + nobs_new, 
                       nobs + 1:nobs + nobs_new, d), SqEuclidean(), view(Xnew, d:d, :))
    end
    setdimension!(kd, nobs + nobs_new, 1:2)
    kerneldata
end

function ElasticKernelData(k::LinArd, X1::AbstractMatrix, X2::AbstractMatrix; capacity = 10^3, stepsize = 10^3)
    @assert X1==X2 # TODO: extend to X1 != X2
    X = X1
    dim, nobs = size(X)
    XtX_d = AllElasticArray(3; capacity = (capacity, capacity, size(X, 1)),
                                    stepsize = (stepsize, stepsize, 0))
    @inbounds @simd for d in 1:dim
        for i in 1:nobs
            for j in 1:i
                XtX_d[i, j, d] = XtX_d[j, i, d] = X[d, i] * X[d, j]
            end
        end
    end
    setdimension!(XtX_d, nobs, 1:2)
    setdimension!(XtX_d, dim, 3)
    LinArdData(XtX_d)
end
function append!(kerneldata::LinArdData, kernel::LinArd, X::AbstractMatrix, Xnew::AbstractMatrix)
    kd, dim, nobs, nobs_new = prepareappend!(kerneldata.XtX_d, Xnew)
    @inbounds @simd for d in 1:dim
        for i in 1:nobs
            for j in 1:nobs_new
                kd[i, nobs + j, d] = kd[nobs + j, i, d] = X[d, i] * Xnew[d, j]
            end
        end
        for i in 1:nobs_new
            for j in 1:i
                kd[nobs + i, nobs + j, d] = kd[nobs + j, nobs + i, d] = Xnew[d, i] * Xnew[d, j]
            end
        end
    end
    setdimension!(kd, nobs + nobs_new, 1:2)
    kerneldata
end

function ElasticKernelData(k::LinIso, X1::AbstractMatrix, X2::AbstractMatrix; capacity = 10^3, stepsize = 10^3)
    @assert X1==X2 # TODO: extend to X1 != X2
    X = X1
    dim, nobs = size(X)
    XtX = AllElasticArray(2, capacity = (capacity, capacity), stepsize = (stepsize, stepsize))
    @inbounds @simd for d in 1:dim
        for i in 1:nobs
            for j in 1:i
                XtX[i, j] = XtX[j, i] += X[d, i] * X[d, j]
            end
        end
    end
    setdimension!(XtX, nobs, 1:2)
    LinIsoData(XtX)
end
function append!(kerneldata::LinIsoData, kernel::LinIso, X::AbstractMatrix, Xnew::AbstractMatrix)
    kd, dim, nobs, nobs_new = prepareappend!(kerneldata.XtX, Xnew)
    @inbounds @simd for d in 1:dim
        for i in 1:nobs
            for j in 1:nobs_new
                kd[i, j + nobs] = kd[j + nobs, i] += X[d, i] * Xnew[d, j]
            end
        end
        for i in 1:nobs_new
            for j in 1:i
                kd[i + nobs, j + nobs] = kd[j + nobs, i + nobs] += Xnew[d, i] * Xnew[d, j]
            end
        end
    end
    setdimension!(kd, nobs + nobs_new, 1:2)
    kerneldata
end
