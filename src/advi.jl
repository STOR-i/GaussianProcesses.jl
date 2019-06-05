using Distributions, GaussianProcesses, Random, LinearAlgebra, Bijectors, PDMats

"""
Run Automatic Differentiation Variational Inference (ADVI) for a supplied GP with
non-Gaussian likelihood function. Full details of the ADVI framework can be found
in
"""
advi = function(gp::GPBase, M::Int, threshold::Float64=0.01, iters::Int=100,
    mc_samples::Int=100, iter_skip::Int=200, max_errors::Int=10)
    # Deepcopy of original GP for updating the target and likelihod while keeping the original GP in tact
    # gp_update = deepcopy(gp)
    pars = grab_params(gp)
    nobs = size(gp.x, 2)

    # Transform the parameters to all exist on the real line
    K = length(pars)
    println(K, " parameters present")

    # Initialise Q's mean vector
    qμ = Array{Float64}(undef, (iters), K)
    qμ[1,:] = zeros(K)

    # Initialise Q's log standard deviation
    qω = Array{Float64}(undef, (iters), K)
    qω[1,:] = zeros(K)

    # Intialise Step-Size matrix and surrogate gradient vectors
    ρ = Array{Float64}(undef, iters, 2*K)
    s = Array{Float64}(undef, iters, 2*K)

    # Initiate ELBO delta
    elbo = 0
    s = 0
    delta = Inf
    nit = 1

#     while delta > threshold
    while nit < iters
        mc_errors = 0
        println("")
        println("Iteration: ", nit)
        # Sample M η from a multivariate Gaussian
        # TODO: Check ADVI's dependency on this and possibly remove - should  be accounted for individual functions
        η = rand(Normal(), K)
        global μ = qμ[nit,:]
        global ω = qω[nit,:]

        # Compute ∇μL from Equation 7
        params_kwargs = GaussianProcesses.get_params_kwargs(gp; domean=true, kern=true, noise=true, lik=true)
        θ_cur = GaussianProcesses.get_params(gp; params_kwargs...)

        # Sample index values to be used for stochastically sampling (x, y) pairs
        # Compute the gradient of the ELBO w.r.t the variational mean parameter μ and the transformed variance ω
        stoch_idx = rand(1:nobs)
        x, y = gp.x[:, stoch_idx], gp.y[stoch_idx]
        ∇μ, ∇ω = elbo_grad(μ, ω, x, y, gp)

        # Check for NaNs or Infs, as in the Stan code
        if count(isfinite.(∇μ)) == length(∇μ) & count(isfinite.(∇μ)) == length(∇μ)
            # Update the step size
            fullρ, s = updateρ(vcat(∇μ, ∇ω), s, nit)
            newρ = reshape(fullρ, 1, 2*K)
            ρ[nit,:] = newρ

            # Update parameters
            qμ[nit+1, :] = update_param(μ, ρ[nit, 1:K], ∇μ)
            qω[nit+1, :] = update_param(ω, ρ[nit, (K+1):end], ∇ω)

            # Stop infinite loop - remove when testing in full
            delta = 0
            # Evaluate ELBO
            elbo = elbo_evaluate(gp, qμ[nit+1, :], qω[nit+1, :], x, y)
            nit += 1
        else
            mc_errors += 1
            if mc_errors > max_errors
                throw(DomainError("Max number of errors exceeded."))
            end
        end

    end
    return qμ, qω
end

"""
Compute the ELBO function for our given model p(θ|x) and the approximating
variational distribution q(ζ).
"""
elbo_evaluate = function(gp, μ, ω, x, y, n_samples::Int=100)
    @assert length(μ) == length(ω)
    elbo_sample = zeros(n_samples)
    for i in 1:n_samples
        η = rand(Normal(), length(μ))
        # println(η)
        # println("μ: ", μ, "   ω: ", ω, "    x: ", x, "    y: ", y)
        new_elbo = joint_exp(gp, η, μ, ω, x, y)
        elbo_sample[i] = new_elbo
    end
    # TODO: Check dimension on which the mean is being computed
    mean(elbo_sample) .+ mean_field_entropy(ω)
end

"""
Compute the original model p's contribtuion to the ELBO
"""
# Firt part of the ELBO equation
joint_exp = function(gp, η, μ, ω, x, y)
    ηt = ηtransform(η, μ, ω)
    return update_ll(gp, ηt, x, y)
end

"""
Elliptically standardise the variational Gaussian distribution into a standard Gaussian.
For a mean-field approximation, this computes
``S^{-1}(η) = σ*(ζ+μ) = exp(ω)*η+μ``

# Arguments:
- `η`: K-dimensional draw from the standard Gaussian
- `μ`: Current variational mean vector
- `ω`: Current variational variance vector
"""
ηtransform(η, μ, ω) = (exp.(ω) .* η) .+ μ


"""
Entropy term of the variational approximation Q. Assuming a Gaussian, the entropy is
``0.5 * dim * (1+log(2π)) + 0.5*log det diag (σ^2)``
Under the transformation of σ to exist in the real coordinate space, the entropy
then becomes
``0.5 * dim * (1 + log(2π)) + ∑ω``
"""
mean_field_entropy(ω) = 0.5 * length(ω) * (1+log(2*pi)) * sum(ω)


"""
Update the GP's likelihood
"""
update_ll = function(gp, η, x, y)
    tempμ = mean(gp.mean, to1d(x.*η))
    F = unwhiten(gp.cK, gp.v) .+ tempμ
    ll = sum(GaussianProcesses.log_dens(gp.lik, F, gp.y))
    return ll
end

"""
Compute the derivative of the ELBO with respect to μ and ω.
"""
# TODO: Expand above documentation
# TODO: Adapt to represent just a single sample, not the entire dataset
elbo_grad = function(μ, ω, x, y, gp)
    @assert length(ω) == length(μ)
    # Draw from standard Gaussian and transform to live in ℝ
    η = rand(Normal(), length(ω))
    ζ = ηtransform(η, μ, ω)

    # Think a new GP will need to be fitted here for each iteration as the GP is not for the full dataset, instead a single MC sample x∼N(0,1)
    ps = GaussianProcesses.get_params_kwargs(gp; domean=true, kern=true, noise=true, lik=true)
    θcur = GaussianProcesses.get_params(gp; ps...)
    calc_target(gp, θcur)
    ∇θp = gp.dtarget[(gp.nobs+1):end]

    # Again, the following works for log transform
    detJ = η

    # Return derivatives
    ∇μ = (∇θp.*ζ) + detJ

    int = ((∇θp.*ζ).+detJ) .* η .* exp.(ω)
    ∇ω = reshape(int, 1, length(int)) .+ ones(1, length(η))
    return reshape(∇μ, 1, length(∇μ)), ∇ω
end



# TODO: Clean up this in the code
"""
Ensure arrays are of correct dim and coerce if necessary.
"""
to2d(param) = reshape(param, 1, length(param))
to1d(param) = reshape(param, length(param))

"""
Update the learning rate, considering previous the gradient vector of previous
iterations, should one exist. This is equivalent to the Robbins-Monroe learning
rate update equation.
"""
# TODO: Expand above documentation
updateρ = function(g, s, nit::Int, ηstep::Float64=0.1, τ::Float64=1.0, α::Float64=0.1, ϵ::Float64=1e-16)
    # TODO: Implement some scale factor warm up
    if nit == 1
        s = g.^2
    else
        s = α.*g.^2 + (1-α).*s # TODO: Incorporate previous time step's gradient information. e.g. αg^2+(0.9)*s[nit-1]
    end
    ρ = ηstep * nit^(-0.5 + ϵ)  * (τ .+ sqrt.(s)).^(-1)
    return ρ, s
end

"""
Update the variational parameters
"""
update_param(param, ρ, deriv) = to2d(param) .+ to2d(ρ) .* deriv


"""
Extract the latent variables from the GP
"""
#TODO: Possible duplication from original package
grab_params = function(gp::GPBase)
    θ = []
    try
        θ = vcat(θ, gp.kernel.priors)
    catch err
        θ = θ
    end
    try
        θ = vcat(θ, gp.lik.priors)
    catch err
        θ = θ
    end
    try
        θ = vcat(θ, gp.mean.priors)
    catch err
        θ = θ
    end
#     θ = vcat(θlik, θkern, θlik)
end

"""
Calculate the target of the GP and its respective derivative.
"""
function calc_target(gp::GPBase, θ::AbstractVector) #log-target and its gradient
    Kgrad = Array{Float64}(undef, gp.nobs, gp.nobs)
    L_bar = Array{Float64}(undef, gp.nobs, gp.nobs)
    params_kwargs = GaussianProcesses.get_params_kwargs(gp; domean=true, kern=true, noise=true, lik=true)
    try
        set_params!(gp, θ; params_kwargs...)
        GaussianProcesses.update_target_and_dtarget!(gp, Kgrad, L_bar; params_kwargs...)
        return true
    catch err
#         if !all(isfinite.(θ))
#             return false
        if isa(err, ArgumentError)
            return false
        elseif isa(err, LinearAlgebra.PosDefException)
            return false
        else
            throw(err)
        end
    end
end

#Simulate the data
Random.seed!(123)
n = 20
X = collect(range(-3,stop=3,length=n));
f = 2*cos.(2*X);
Y = [rand(Poisson(exp.(f[i]))) for i in 1:n];

#GP set-up
k = Matern(3/2,0.0,0.0)   # Matern 3/2 kernel
l = PoisLik()
m = MeanConst(0.2)

gp = GP(X, vec(Y), m, k, l)

set_priors!(gp.kernel,[Normal(-2.0,4.0),Normal(-2.0,4.0)])
# set_priors!(gp.lik,[Weibull(1.5, 1)])
set_priors!(gp.mean, [Normal(0, 2.0)])

qμ, qω = advi(gp, 10, 2.0, 100)
println(qμ)
println(qω)
