function initialise_Q(gp::GPBase)
    # TODO: Use PDMats for the below
    V = cov(gp.kernel, gp.x, gp.data)
    Ω = inv(V)
    K = deepcopy(Ω)
    m = mean(gp.mean, gp.x)
    Q = Approx(m, V)
    return Q, V, K
end


function update_Q!(Q, m, V)
    Q.m = m
    Q.V = V
end


function elbo(y::AbstractArray, μ::AbstractArray, Ω::AbstractMatrix, m::AbstractArray, V::AbstractMatrix, ll::Likelihood)
    @assert length(μ) == length(m)
    @assert size(V) == size(Ω)
    println("y: ", size(y))
    println("μ: ", size(μ))
    println("Ω: ", size(Ω))
    println("m: ", size(m))
    println("V: ", size(V))
    N = length(y)
    VprodΩ = V * Ω
    vexp =  var_exp(ll, y, m, V)
    return 0.5*(logdet(VprodΩ) - tr(VprodΩ) - transpose(m - μ)*Ω*(m - μ) + N) + vexp
end


function push_back(mat::AbstractMatrix, idx::Integer)
    # TODO: Possibly more efficient way to do this.
    ori_size = size(mat)
    target = mat[:, idx]
    temp =  mat[:,setdiff(1:end, idx)]
    mat = cat(temp, target, dims=2)
    return mat
end


function push_back(mat::AbstractVector)
    target = mat[1]
    append!(mat[2:end], target)
    return mat
end


function vi(gp::GPBase; nits::Int64=100)
    function objective_func(params)
        n = gp.nobs
        m = params[1:n]
        V = params[(n + 1):end]
        prod_term = V .* ω
        obj = -0.5*(logdet(prod_term .* Array{Float64}(I, n, n)) - sum(prod_term) - ((m - gp.μ)' * Ω * (m - gp.μ)) + n + var_exp(gp.lik, gp.y, m, V))
        return obj
    end

    # Initialise log-target and log-target's derivative
    mcmc(gp; nIter=1);

    # Initialise variational approximation
    Q, Ω, K = initialise_Q(gp)
    ω = diag(Ω)

    θ = cat(Q.m, diag(Q.V), dims=1)

    # res = optimize(objective_func, θ, Newton())# LBFGS(); autodiff = :forward)
    lower = cat(repeat([-Inf], gp.nobs), repeat([1e-10], gp.nobs), dims = 1)
    upper = cat(repeat([Inf], gp.nobs), repeat([Inf], gp.nobs), dims = 1)
    initial_x = cat(Q.m, diag(Q.V), dims=1)
    inner_optimizer = ConjugateGradient()
    res = optimize(objective_func, lower, upper, initial_x, Fminbox(inner_optimizer))

    optθ = Optim.minimizer(res)
    Q.m = optθ[1:gp.nobs]
    Q.V = (optθ[(gp.nobs + 1):end] .* Array{Float64}(I, gp.nobs, gp.nobs))  + Array{Float64}(I, gp.nobs, gp.nobs)*1e-10

    return Q
end

function predict_f(gp::GPBase, x::AbstractMatrix, Q::Approx)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
    return predict_full(gp, x, Q)
end

# Sample from the variational GP
function Random.rand!(gp::GPBase, x::AbstractMatrix, A::DenseMatrix, Q::Approx)
    nobs = size(x, 2)
    n_sample = size(A, 2)
    μ, Σraw = predict_f(gp, x, Q)
    Σraw, chol = make_posdef!(Σraw)
    Σ = PDMat(Σraw, chol)
    return broadcast!(+, A, μ, unwhiten!(Σ,randn(nobs, n_sample)))
end

# Generate samples from the variational approxiamtion
Random.rand(gp::GPBase, x::AbstractMatrix, n::Int, Q::Approx) = rand!(gp, x, Array{Float64}(undef, size(x, 2), n), Q)
Random.rand(gp::GPBase, x::AbstractVector, Q::Approx) = vec(rand(gp, x', 1, Q))
Random.rand(gp::GPBase, x::AbstractMatrix, Q::Approx) = vec(rand(gp, x, 1, Q))

predict_full(gp::GPA, xpred::AbstractMatrix, Q::Approx) = predictMVN(gp,xpred, gp.x, gp.y, gp.kernel, gp.mean, gp.v, gp.covstrat, Q)

"""
    predict_y(gp::GPA, x::Union{Vector{Float64},Matrix{Float64}}[; full_cov::Bool=false])
Return the predictive mean and variance of Gaussian Process `gp` at specfic points which
are given as columns of matrix `x`. If `full_cov` is `true`, the full covariance matrix is
returned instead of only variances.
"""
function predictMVN(gp::GPBase,xpred::AbstractMatrix, xtrain::AbstractMatrix, ytrain::AbstractVector,
                   kernel::Kernel, meanf::Mean, alpha::AbstractVector,
                   covstrat::CovarianceStrategy, Q::Approx)
    Ktrain = PDMat(Q.V)
    crossdata = KernelData(kernel, xtrain, xpred)
    priordata = KernelData(kernel, xpred, xpred)
    Kcross = cov(kernel, xtrain, xpred, crossdata)
    Kpred = cov(kernel, xpred, xpred, priordata)
    mx = mean(meanf, xpred)
    mu, Sigma_raw = predictMVNvi!(gp,Kpred, Ktrain, Kcross, mx, alpha) #TODO: Handle through multiple dispatch
    return mu, Sigma_raw
end

function predictMVN!(gp::GPA, Kxx, Kff, Kfx, mx, αf)
    Lck = whiten!(Kff, Kfx)
    mu = mx + Lck' * αf
    subtract_Lck!(Kxx, Lck)
    return mu, Kxx
end

function predictMVNvi!(gp::GPA, Kxx, Kff, Kfx, mx, αf)
    Lck = whiten!(Kff, Kfx)
    mu = mx + Lck' * αf
    return mu, Kxx
end

