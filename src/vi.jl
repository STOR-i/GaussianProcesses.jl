mutable struct Approximating
    qμ::Float64
    q⁠Σ::Float64
end

function vi(gp::GPBase; threshold::Int=1)
    print(elbo(gp))
end

function elbo(gp::GPBase)
    q = Approximating(gp.nobs, gp.nobs)

    kl = 0.5(dot(q.qμ, q.qμ) - logdet(q.qΣ) + sum(diag(q.qΣ).^2)) #KL prior gives the divergence between two Gaussians

    gp.μ = mean(gp.m, gp.X)          #mean function
    Σ = cov(gp.k, gp.X, gp.data)    #kernel function
    gp.cK = PDMat(Σ + 1e-6*I)
    Fmean = unwhiten(gp.cK, q.qμ) + q.μ      # K⁻¹q_μ
    Fvar = unwhiten(gp.cK, q.qΣ)              # K⁻¹q_Σ
    varExp = var_exp(gp.lik, Fmean, Fvar)      # ∫log p(y|f)q(f), where q(f) is a Gaussian approx.
    gp.ll = varExp - KL # Log-likelihood lower bound
end
