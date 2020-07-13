abstract type GP end

struct GPR{A<:MeanFunction, B<:Kernel} <: GP
    MeanFunc::A
    Kernel::B
end
GPR{A, B}(mean_function, kernel) where {A, B} = GPR(mean_function, kernel)
# Assume a zero-mean function if a GP constructor is called with only a kernel
GPR(kernel::Kernel) = GPR(Zero(), kernel)

function marginal_log_likelihood(gp::GPR, x::AbstractArray, y::AbstractArray)
    N = size(x, 2) # TODO: Change if moving to column vectors
    K = cov(gp.Kernel, x) + 1e-6*I(N)
    L = cholesky(K)
    α = L.U\(L.L\y')
    mll = -0.5*((y*α)[1]+ 2*sum(diag(L.UL)) - N*log(2π))
end

function get_mll(gp::GPR, x::AbstractArray, y::AbstractArray)
    function mll(x::AbstractArray, y::AbstractArray)
        N = size(x, 2)
        # TODO: Return function mll
    end 
    return mll 
end


"""
Sample from the GP prior

# Examples
```jldoctest
julia> x = cat(collect(-1:0.1:1), dims=2)'
julia> f = GP(Zero(), SquaredExponential(0.5, 1.0))
julia> rand(f, x; n_samples=10)
```
"""
function Base.rand(gp::GPR, x::AbstractArray; n_samples::Int=1)
    n = length(vec(x))
    μ = mean(gp.MeanFunc, x)
    Σ = cov(gp.Kernel, x) + 1e-6*I(n)
    dist = MultivariateNormal(vec(μ), Σ) # TODO: Return a vector from mean function
    return rand(dist, n_samples)
end

# x = cat(collect(-1:0.1:1), dims=2)'   
# y = sin.(x)
# f = GPR(SquaredExponential(0.1, 1.0))
