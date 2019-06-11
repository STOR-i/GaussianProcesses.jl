using GaussianProcesses, RDatasets, LinearAlgebra, Statistics, PDMats, Optim, ForwardDiff
import Distributions:Normal, Poisson
import GaussianProcesses: predict_obs, get_params_kwargs, get_params
using Random
using Optim


mutable struct Approx
    qμ
    qΣ
end

function update_Q!(Q::Approx, params::Array)
    Q.qμ = params[1]
    Q.qΣ = params[2]
end

function vector_hessian(f, x)
       n = length(x)
       out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
       return reshape(out, n, n, n)
   end

abstract type AbstractGradientPrecompute end

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

init_precompute(gp::GPMC) = FullCovMCMCPrecompute(gp.nobs)


function elbo_grad(gp::GPBase)
   # TODO: νbar is the loglikelihoods derivatives w.r.t each parameter. What is plain ν
   # TODO: Can ν just equal νbar, as per Section 4?
   # TODO: Get derivs of ll here
   ν = gp.dll
   B = gp.
   # Computing EQ16 of Opper
   ∇θ = -0.5*(ν * tr(ν) .- inv(gp.cK))
   print(∇θ)
end

# """
# Set the GP's posterior distribution to be the multivariate Gaussian approximation.
# """
# function approximate!(gp::GPBase, Q::Approx)
# end

function vi(gp::GPBase; verbose::Bool=false, nits::Int=100)
    # Initialise the varaitaional parameters
    Q = Approx(zeros(gp.nobs), Matrix(I, gp.nobs, gp.nobs)*1.0)
    # Compute the initial ELBO objective between the intiialised Q and the GP
    λ = [zeros(gp.nobs), Matrix(I, gp.nobs, gp.nobs)*1.0]

    function elbo(params)
        # Compute the prior KL
        update_Q!(Q, params)
        kl = 0.5(dot(Q.qμ, Q.qμ) - logdet(Q.qΣ) + sum(diag(Q.qΣ).^2))
        gp.μ = mean(gp.mean, gp.x)
        Σ= cov(gp.kernel, gp.x, gp.data)    #kernel function
        gp.cK = PDMat(Σ + 1e-6*I)
        Fmean = unwhiten(gp.cK, Q.qμ) + gp.μ      # K⁻¹q_μ

        # Assuming a mean-field approximation
        Fvar = diag(unwhiten(gp.cK, Q.qΣ))              # K⁻¹q_Σ
        _, varExp = predict_obs(gp.lik, Fmean, Fvar)      # ∫log p(y|f)q(f), where q(f) is a Gaussian approx.
        # ELBO = Σ_n 𝔼_{q(f_n)} ln p(y_n|f_n) + KL(q(f)||p(f))
        return sum(varExp) - kl
    end

    # Compute dll
    precomp = init_precompute(gp)
    params_kwargs = get_params_kwargs(gp; domean=true, kern=true, noise=true, lik=true)
    print(params_kwargs)
    count = 0
    function calc_target(gp::GPBase, θ::AbstractVector) #log-target and its gradient
        count += 1
        set_params!(gp, θ; params_kwargs...)
        println(gp)
        GaussianProcesses.update_target_and_dtarget!(gp, precomp; params_kwargs...)
    end

    θ_cur = get_params(gp; params_kwargs...)
    D = length(θ_cur)
    calc_target(gp, θ_cur)

    elbo(λ)
    elbo_grad(gp)
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

vi(gp)
# set_priors!(gp.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0)])
# set_priors!(gp.mean, [Normal(0, 2.0)])
