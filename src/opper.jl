using GaussianProcesses, RDatasets, LinearAlgebra, Statistics, PDMats, Optim, ForwardDiff, Plots
import Distributions:Normal, Poisson
import GaussianProcesses: predict_obs, get_params_kwargs, get_params, predict_f, update_ll_and_dll!, optimize!
using Random
using Optim


mutable struct Approx
    qμ
    qΣ
end


abstract type AbstractGradientPrecompute end
abstract type CovarianceStrategy end
struct FullCovariance <: CovarianceStrategy end

struct FullCovMCMCPrecompute <: AbstractGradientPrecompute
    L_bar::Matrix{Float64}
    dl_df::Vector{Float64}
    f::Vector{Float64}
end

function FullCovMCMCPrecompute(nobs::Int)
    buffer1 = Matrix{Float64}(undef, nobs, nobs)
    buffer2 = Vector{Float64}(undef, nobs)
    buffer3 = Vector{Float64}(undef, nobs)
    return FullCovMCMCPrecompute(buffer1, buffer2, buffer3)    
end

function init_precompute(covstrat::FullCovariance, X, y, k)
    nobs = size(X, 2)
    FullCovariancePrecompute(nobs)
end

init_precompute(gp::GPMC) = FullCovMCMCPrecompute(gp.nobs)

function update_Q!(Q::Approx, params::Array)
    Q.qμ = params[1]
    Q.qΣ = params[2]
end


# Possibly unnecessary function for optim
function vector_hessian(f, x)
       n = length(x)
       out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
       return reshape(out, n, n, n)
   end


# Compute the Hadamard product
function hadamard(A::Matrix, B::Matrix)
    @assert size(A) == size(B)
    H = Array{Float64}(undef, size(A)) 
    n, m = size(A)
    for j in 1:n
        for i in 1:m
            H[i,j] = A[i, j] * B[i,j]
        end
    end
    return H
end


"""
Compute Σ using the Sherman-Woodbury-Morrison identity, such that Σ=[K^{-1}+Λ^2]^{-1} = Λ^{-2} - Λ^{-1}A^{-1}Λ^{-1} where A=ΛKΛ + I, such that K is the GP's covariance matrix and Λ=diag(λ) where λ is our variational approximation's variance parameters.
"""
# TODO: Remove Q when qμ and qΣ are incorporated into GPMC
function computeΣ(gp::GPBase, Q::Approx)
    Λ = diag(Q.qΣ) .* Matrix(I, size(Q.qΣ, 1), size(Q.qΣ, 1))
    A = (Λ .* gp.cK.mat .* Λ) .+ (Matrix(I, gp.nobs, gp.nobs)*1.0)
    Σ = Λ.^(-2) .- (Λ^(-1) .* A^(-1) .* Λ^(-1))
    return Σ
end


"""
Compute the gradient of the ELBO F w.r.t. the variational parameters μ and Σ, as per Equations (11) and (12) in Opper and Archambeau.
"""
function elbo_grad_q(gp::GPBase, Q::Approx)
    νbar = -gp.dll[1:gp.nobs]
    gν = gp.cK.mat*(Q.qμ - νbar) # TODO: Should this be a product of the application of the covariance function to ν-νbar?
    Σ = computeΣ(gp, Q)
    λ = Q.qΣ
    # λbar = 
    gλ = diag(0.5*(hadamard(Σ, Σ))) # Must multiply by λ-λbar
    return gν, gλ
end


function elbo_grad_θ(gp::GPBase)
   # TODO: Can ν just equal νbar, as per Section 4?
   νbar = gp.dll[1:gp.nobs]
   
   # Computing EQ16 of Opper
   ∇θ = -0.5*(dot(νbar, νbar) .- inv(gp.cK.mat))
   print(∇θ)
end


"""
Update the parameters of the variational approximation through gradient ascent
"""
function updateQ!(Q::Approx, ∇μ, ∇Σ; α::Float64=0.01)
    Q.qμ += α*-∇μ
    Q.qΣ += α*-∇Σ .* (Matrix(I, length(∇Σ), length(∇Σ)) *1.0)
end


"""
Set the GP's posterior distribution to be the multivariate Gaussian approximation.
"""
function approximate!(gp::GPBase, Q::Approx)
end


"""
Carry out variational inference, as per Opper and Archambeau (2009) to compute the GP's posterior, given a non-Gaussian likelihood.
"""
function vi(gp::GPBase; verbose::Bool=false, nits::Int=100, plot_elbo::Bool=false)
    # Initialise log-target and log-target's derivative
    mcmc(gp; nIter=1)
    optimize!(gp)
    Q = Approx(gp.v, Matrix(I, gp.nobs, gp.nobs)*1.0)
    # Initialise the varaitaional parameters
#    Q = Approx(zeros(gp.nobs), Matrix(I, gp.nobs, gp.nobs)*1.0)
    # Compute the initial ELBO objective between the intiialised Q and the GP
    λ = [zeros(gp.nobs), Matrix(I, gp.nobs, gp.nobs)*1.0]
    

    function elbo(params)
        # Compute the prior KL e.g. KL(Q||P) s.t. P∼N(0, I)
        kl = 0.5(dot(Q.qμ, Q.qμ) - logdet(Q.qΣ) + sum(diag(Q.qΣ).^2))
        @assert kl >= 0 "KL-divergence should be positive.\n"
        println("KL: ", kl)
        μ = mean(gp.mean, gp.x)
        Σ= cov(gp.kernel, gp.x, gp.data)    #kernel function
        gp.cK = PDMat(Σ + 1e-6*I)
        Fmean = unwhiten(gp.cK, Q.qμ) + μ      # K⁻¹q_μ

        # Assuming a mean-field approximation
        Fvar = diag(unwhiten(gp.cK, Q.qΣ))              # K⁻¹q_Σ
        _, varExp = predict_obs(gp.lik, Fmean, Fvar)      # ∫log p(y|f)q(f), where q(f) is a Gaussian approx.
        # ELBO = Σ_n 𝔼_{q(f_n)} ln p(y_n|f_n) + KL(q(f)||p(f))
        elbo_val = sum(varExp)-kl
        println("ELBO: ", elbo_val)
        # @assert elbo_val <= 0 "ELBO Should be less than 0.\n"
        return sum(varExp) - kl
    end
    init_elbo = elbo(λ) # TODO: Change this λ
    if verbose
        println("Initial ELBO: ", init_elbo)
    end
    
    global elbo_approx = Array{Float64}(undef, nits+1)
    elbo_approx[1] = init_elbo

    # Iteratively update variational parameters
    for i in 1:nits
        buff = init_precompute(gp)
        update_ll_and_dll!(gp, buff)
        λ = [Q.qμ, Q.qΣ]

        # Compute the gradients of the variational objective function
        gradμ, gradΣ = elbo_grad_q(gp, Q)

        # Update the variational parameters
        updateQ!(Q, gradμ, gradΣ)

        # Recalculate the ELBO
        λ = [Q.qμ, Q.qΣ]
        current_elbo = elbo(λ)
        elbo_approx[i+1] = current_elbo

        if verbose
            println("ELBO at Iteration ", i, ": ", current_elbo)
        end
    end

    if plot_elbo
        println(elbo_approx)
        # plot(0:nits, elbo_approx)
    end
end


Random.seed!(123)

n = 20
X = collect(range(-3,stop=3,length=n));
f = 2*cos.(2*X);
Y = [rand(Poisson(exp.(f[i]))) for i in 1:n];

#GP set-up
k = Matern(3/2,0.0,0.0)   # Matern 3/2 kernel
l = PoisLik()             # Poisson likelihood

gp = GP(X, vec(Y), MeanZero(), k, l)
set_priors!(gp.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0)])

# mcmc(gp)
vi(gp;nits=10, verbose=true, plot_elbo=true)
