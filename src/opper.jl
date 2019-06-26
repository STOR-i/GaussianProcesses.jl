using GaussianProcesses, RDatasets, LinearAlgebra, Statistics, PDMats, Optim, ForwardDiff, Plots, Calculus
import Distributions:Normal, Poisson
import GaussianProcesses: get_params_kwargs, get_params, predict_f, update_ll_and_dll!, optimize!, update_target_and_dtarget!, gausshermite, log_dens, sqrtπ
using Random
using Optim
import PDMats: unwhiten!

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

# Compute Σ crudely as per Opper and Archambeau
function computeΣ(gp::GPBase, λ::Array)
    return inv(inv(gp.cK.mat) .+ λ)
end

"""
Compute the gradient of the ELBO F w.r.t. the variational parameters μ and Σ, as per Equations (11) and (12) in Opper and Archambeau.
"""
function elbo_grad_q(gp::GPBase, Q::Approx)
    νbar = -gp.dll[1:gp.nobs]
    gν = gp.cK.mat*(Q.qμ - νbar) # TODO: Should this be a product of the application of the covariance function to ν-νbar?
    Σ = computeΣ(gp, diag(Q.qΣ))
    λ = Q.qΣ
    λbar = -gp.dll[1:gp.nobs] .* (Matrix(I, gp.nobs, gp.nobs)*1.0)
    gλ = diag(0.5*(hadamard(Σ, Σ) .* (λ - λbar)))
    return gν, gλ
end

"""
Compute the gradient of the ELBO F w.r.t. the variational parameters μ and Σ using Julia Math's numerical approximation.
"""
function elbo_grad_q_numerical(gp, qμ::AbstractArray, qΣ::AbstractMatrix)
    params = qμ
    # Numerical approximation (just looking at Q.qμ)
    μ_grad = Calculus.gradient(params) do params
        qμ = params
        elbo(gp, Q)
    end

    params = diag(qΣ)
    Σ_grad = Calculus.gradient(params) do params
        qΣ = params
        elbo(gp, Q)
    end

    return μ_grad, Σ_grad
end

function elbo_grad_q_numerical(gp, qμ::AbstractArray, qΣ::AbstractArray)
    params = qμ
    # Numerical approximation (just looking at Q.qμ)
    μ_grad = Calculus.gradient(params) do params
        qμ = params
        elbo(gp, Q)
    end

    params = qΣ
    Σ_grad = Calculus.gradient(params) do params
        qΣ = params
        elbo(gp, Q)
    end

    return μ_grad, log(Σ_grad)
end


# Compute gradient of the ELBO w.r.t the GP's kernel parameters
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
function updateQ!(Q::Approx, ∇μ::AbstractArray, ∇Σ::AbstractMatrix; α::Float64=0.001)
    Q.qμ += α*-∇μ
#    Q.qΣ += α*-diag((∇Σ .* (Matrix(I, length(∇Σ), length(∇Σ)) *1.0))) #need to stop parameters becoming negative
end

function updateQ!(Q::Approx, ∇μ::AbstractArray, ∇Σ::AbstractArray; α::Float64=0.001)
#    Q.qΣ = ∇Σ .* Matrix{Float64}(I, length(∇μ), length(∇μ))*1.0
    Q.qμ += α*-∇μ
#    Q.qΣ += α*-(∇Σ .* (Matrix(I, length(∇Σ), length(∇Σ)) *1.0)) #need to stop parameters becoming negative
end

"""
Update only the variational mean.
"""
function updateQ!(Q::Approx, ∇μ::AbstractArray; α::Float64=0.01)
    Q.qμ += α*-∇μ
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

    # TODO: Remove globals
    # Initialise the varaitaional parameters
    global Q = Approx(zeros(gp.nobs), Matrix(I, gp.nobs, gp.nobs)*1.0)
    # Compute the initial ELBO objective between the intiialised Q and the GP
    λ = [zeros(gp.nobs), Matrix(I, gp.nobs, gp.nobs)*1.0]

    # Compute the ELBO function as per Opper and Archambeau EQ (9)
    function elbo(gp, Q)
        μ = mean(gp.mean, gp.x)
        Σ = cov(gp.kernel, gp.x, gp.data)    #kernel function
        L = cholesky(Σ)
        Fmean = L.L * Q.qμ # Assuming a zero mean function. In the case of a non-zero MF, sum this to the product.
        K = PDMat(Σ + 1e-6*I)

        
        # Fmean = unwhiten(K, Q.qμ) + μ      # \sqrt{K}*q_μ
        # # Assuming a mean-field approximation
        # Fvar = unwhiten(K, diag(Q.qΣ))              # \sqrt{K}*q_Σ

        # Fmean_prev = unwhiten(K, Q.qμ) + μ      # K⁻¹q_μ
        # Compute Fvar

        q_sqrt_dnn = LowerTriangular(Q.qΣ)
        L_tiled = L # In the case of multioutput GP, this would need to tiled d times, where d is the output dimension.
        LTA = L_tiled.L * q_sqrt_dnn
        Fvar = transpose(sum(LTA.data, dims=2)) # TODO: When log-transform Q.qΣ, LTA should be exponentiated

        # Assuming a mean-field approximation
        # Fvar = unwhiten(K, qΣexp)              # K⁻¹q_Σ
        varExp = expect_dens(gp.lik, Fmean, Fvar, gp.y)      # ∫log p(y|f)q(f), where q(f) is a Gaussian approx.

        # Compute KL as per Opper and Archambeau eq (9)
        Σopper = computeΣ(gp, Q.qΣ)
        Kinv = inv(K.mat)
        # # Compute the prior KL e.g. KL(Q||P) s.t. P∼N(0, I)
        # kl = 0.5(dot(Q.qμ, Q.qμ) - logdet(Q.qΣ) + sum(diag(Q.qΣ).^2))
        # @assert kl >= 0 "KL-divergence should be positive.\n"
        # println("KL: ", kl)

        kl = 0.5*tr(Q.qΣ * Kinv) .+ 0.5(transpose(Q.qμ-Fmean) * Kinv * (Q.qμ-Fmean)) .+ 0.5(logdet(K.mat)-logdet(Q.qΣ)) - 0.5*gp.nobs #I've made a change to the logdet that I need to check

        # ELBO = Σ_n 𝔼_{q(f_n)} ln p(y_n|f_n) + KL(q(f)||p(f))
        elbo_val = sum(varExp)-kl
        @assert elbo_val <= 0 "ELBO Should be less than 0.\n"
        return -elbo_val
    end

    # Compute the ELBO function as per GPFlow VGP._buill_ll(). Note, this is different from the _build_ll() in VGP_Opper of GPFlow
    function elbo(Q)
        # Compute the prior KL e.g. KL(Q||P) s.t. P∼N(0, I)
        kl = 0.5(dot(Q.qμ, Q.qμ) - logdet(Q.qΣ) + sum(diag(Q.qΣ).^2))
        @assert kl >= 0 "KL-divergence should be positive.\n"
        # println("KL: ", kl)

        # Following block computes K^{-1}q_{μ}
        μ = mean(gp.mean, gp.x)
        Σ =  cov(gp.kernel, gp.x, gp.data)    #kernel function
        K = PDMat(Σ + 1e-6*I)
        Fmean = unwhiten(K, Q.qμ) + μ      # K⁻¹q_μ

        # Assuming a mean-field approximation
        Fvar = unwhiten(K, diag(Q.qΣ))              # K⁻¹q_Σ
        varExp = expect_dens(gp.lik, Fmean, Fvar, gp.y)      # ∫log p(y|f)q(f), where q(f) is a Gaussian approx.

        # ELBO = Σ_n 𝔼_{q(f_n)} ln p(y_n|f_n) + KL(q(f)||p(f))
        elbo_val = sum(varExp)-kl
        # @assert elbo_val <= 0 "ELBO Should be less than 0.\n"
        return elbo_val
    end

    init_elbo = elbo(gp, Q)
    if verbose
        println("Initial ELBO: ", init_elbo)
    end

    elbo_approx = Array{Float64}(undef, nits+1)
    elbo_approx[1] = init_elbo


    # Iteratively update variational parameters
    for i in 1:nits
        # Run the following two lines as a proxy for computing gp.dll
        params_kwargs = get_params_kwargs(gp; domean=true, kern=true, noise=false, lik=true)
        update_target_and_dtarget!(gp; params_kwargs...)

        # Compute the gradients of the variational objective function
        # gradμ, gradΣ = elbo_grad_q_numerical(gp, Q.qμ, Q.qΣ)

        params = Q.qμ

        # TODO: Check the gradients are being correctly computed. If so, check ELBO function.
        gradμ = Calculus.gradient(params) do params
            Q.qμ = params
            elbo(gp, Q)
        end

        # params = diag(Q.qΣ)

        #
        # gradΣ = Calculus.gradient(params) do params
        #     Q.qΣ = Diagonal(params)+zeros(length(params),length(params))
        #     elbo(gp, Q)
        # end

        # Update the variational parameters
        updateQ!(Q, gradμ)
        println("Variational Mean: ", mean(Q.qμ))

        # Recalculate the ELBO
        current_elbo = elbo(gp, Q)
        elbo_approx[i+1] = current_elbo

        if verbose
            println("ELBO at Iteration ", i, ": ", current_elbo, "\n")
        end
    end

    if plot_elbo
        println(elbo_approx)
        # plot(0:nits, elbo_approx)
    end
end

function expect_dens(lik::Likelihood, fmean::AbstractVector, fvar::AbstractVector, y::AbstractVector)
    n_gaussHermite = 20
    nodes, weights = gausshermite(n_gaussHermite)
    weights /= GaussianProcesses.sqrtπ
    f = fmean .+ sqrt.(2*fvar)*nodes'
    lpred = Array{Float64}(undef, size(f));
    @inbounds for i in 1:n_gaussHermite
        fi = view(f, :, i)
        lpred[:,i] = log_dens(lik, fi, y)
    end
    return lpred*weights
end

function expect_dens(lik::Likelihood, fmean::AbstractVector, fvar::AbstractMatrix, y::AbstractVector)
    fvar = diag(fvar)
    n_gaussHermite = 20
    nodes, weights = gausshermite(n_gaussHermite)
    weights /= GaussianProcesses.sqrtπ
    f = fmean .+ sqrt.(2*fvar)*nodes'
    lpred = Array{Float64}(undef, size(f));
    @inbounds for i in 1:n_gaussHermite
        fi = view(f, :, i)
        lpred[:,i] = log_dens(lik, fi, y)
    end
    return lpred*weights
end


Random.seed!(123)

n = 50
X = collect(range(-3,stop=3,length=n));
f = 2*cos.(2*X);
Y = [rand(Poisson(exp.(f[i]))) for i in 1:n];

#GP set-up
k = Matern(3/2,0.0,0.0)   # Matern 3/2 kernel
l = PoisLik()             # Poisson likelihood

gp = GP(X, vec(Y), MeanZero(), k, l)
set_priors!(gp.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0)])

#vi(gp;nits=50, verbose=true, plot_elbo=true)


samples = mcmc(gp; nIter=10000,ε=0.01);

#Sample predicted values
xtest = range(minimum(gp.x),stop=maximum(gp.x),length=50);
ymean = [];
fsamples = Array{Float64}(undef,size(samples,2), length(xtest));
for i in 1:size(samples,2)
    set_params!(gp,samples[:,i])
    update_target!(gp)
    push!(ymean, predict_y(gp,xtest)[1])
    fsamples[i,:] = rand(gp, xtest)
end

using Plots, Distributions
#Predictive plots
q10 = [quantile(fsamples[:,i], 0.1) for i in 1:length(xtest)]
q50 = [quantile(fsamples[:,i], 0.5) for i in 1:length(xtest)]
q90 = [quantile(fsamples[:,i], 0.9) for i in 1:length(xtest)]
plot(xtest,exp.(q50),ribbon=(exp.(q10), exp.(q90)),leg=true, fmt=:png, label="quantiles")
plot!(xtest,mean(ymean), label="posterior mean")
plot!(xtest,visamps,label="VI approx")
xx = range(-3,stop=3,length=1000);
f_xx = 2*cos.(2*xx);
plot!(xx, exp.(f_xx), label="truth")
scatter!(X,Y, label="data")

#
#
# visamps=  rand(gp, xtest)



########################
#Test gradients
########################
num_test = false
if num_test
    #Set the GP
    params_kwargs = get_params_kwargs(gp; domean=true, kern=true, noise=false, lik=true)
    update_target_and_dtarget!(gp; params_kwargs...)

    Q = Approx(randn(gp.nobs), Matrix(I, gp.nobs, gp.nobs)*1.0)

    #Calculate the elbo and its gradient
    elbo(gp, Q)

    # Compute the gradients of the variational objective function for either qμ or qΣ
    exact_grad = elbo_grad_q(gp, Q)[1]

    params = Q.qμ
    # Numerical approximation (just looking at Q.qμ)
    μ_grad = Calculus.gradient(params) do params
        Q.qμ = params
        elbo(gp, Q)
    end

    params = Q.qΣ
    Σ_grad = Calculus.gradient(params) do params
        Q.qΣ = params
        elbo(gp, Q)
    end

    elbo_grad_q_numerical(gp, Q.qμ, Q.qΣ)



num_grad ≈ μ_grad

#########################################################
Q.qμ = mean(fsamples;dims=1)[:]
Q.qΣ = cov(fsamples)

el=elbo(gp,Q)

g(Q) = elbo(gp,Q)
g'(Q)

#Correct expect_dens for the Poisson case according to GPflow
gp.y.*Fmean - exp.(Fmean + Fvar/2) -lgamma.(gp.y.+1.0) + gp.y 
