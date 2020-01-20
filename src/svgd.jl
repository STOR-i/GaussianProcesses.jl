using Distances, Statistics, ProgressBars
"""
    svgd(gp::GPBase; kwargs...)

Runs Stein variational gradient descent to estimate the posterior distribution of the hyperparameters of Gaussian process `GPE` and the latent function in the case of `GPA`.
"""
function svgd(gp::GPBase; nIter::Int=1000, nParticles::Int = 10, ε::Float64=0.1,
              bandwidth::Float64=-1.0, α::Float64 = 0.9, lik::Bool=true,
              noise::Bool=true, domean::Bool=true, kern::Bool=true,
              trace::Bool=true)

    function calc_dtarget!(gp::GPBase, θ::AbstractVector) #log-target and its gradient
        set_params!(gp, θ; params_kwargs...)
        update_target_and_dtarget!(gp; params_kwargs...)
        pass = true
        if !all(isfinite.(gp.dtarget))
            pass = false
        end
        return pass
    end

    function svgd_kernel(θ::Matrix{Float64};h::Float64=-1.0)  # function to calculate the kernel
        pairwise_dist = pairwise(Euclidean(),θ,dims=2).^2  #check that the squared term is correct
        if h<0
            h = median(pairwise_dist)
            h = sqrt(0.5*h/log(size(θ,1)+1))
        end
        #compute the kernel
        Kxy = exp.(-pairwise_dist/(2*h^2))
        dxkxy = -Kxy*θ'
        sumkxy = sum(Kxy; dims=1)
        for i in 1:size(θ,1)
            dxkxy[:,i] = dxkxy[:,i] + θ[i,:].*vec(sumkxy)
        end
        dxkxy = dxkxy/(h^2)
        return Kxy, dxkxy
    end

    #optimize!(gp)      #find the MAP/MLE as a starting point
    params_kwargs = get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)

    θ_cur = get_params(gp; params_kwargs...)
    D = length(θ_cur)
    θ_particles =   randn(D,nParticles)   #set-up the particles - might be better to sample from the priors
    grad_particles = zeros(D,nParticles)           #set-up the gradients

    fudge_factor = 1e-6   #this is for adagrad
    historical_grad = 0

    for t in ProgressBar(1:nIter)
        for i in 1:nParticles
            while !calc_dtarget!(gp,θ_particles[:,i]) #this is a bit of a hack at the minute and need to think of a better solution. Essentially, if calculating the gradient at one of the particles leads to an instability, then we reset that particle. There's probably a better way of solving this and it's likely and initialisation issue.
                θ_particles[:,i] = randn(D,1)
            end
            grad_particles[:,i] = gp.dtarget
        end
        kxy, dxkxy = svgd_kernel(θ_particles; h = bandwidth)
        grad_θ = (kxy*grad_particles' + dxkxy) / nParticles

        #adagrad
        if t==1
            historical_grad = historical_grad .+ grad_θ.^2
        else
            historical_grad = α*historical_grad .+ (1-α)*grad_θ.^2
        end
        new_grad = grad_θ#./(fudge_factor .+ sqrt.(historical_grad))
        θ_particles += ε*new_grad'

        if LinearAlgebra.norm(grad_θ)<10e-6  #early stopping if converged
            println("Converged at iteration ", t)
            break
        end
    end
    return θ_particles
end
