# using Distances, Statistics, ProgressBars
# import GaussianProcesses: get_params_kwargs, get_params, update_target_and_dtarget, update_target, update_Q
"""
    svgd(gp::GPBase; kwargs...)

Runs Stein variational gradient descent to estimate the posterior distribution of the hyperparameters of Gaussian process `GPE` and the latent function in the case of `GPA`.
"""
rbf(τ::Float64; ℓ::Float64=0.1, σ::Float64=1.0) = exp(-τ/(2*ℓ^2))
rbf(x::Float64, y::Float64; ℓ::Float64=0.1, σ::Float64=1.0) = exp(-abs(x-y)/(2*ℓ^2))

function svgd(gp::GPBase; nIter::Int=1000, nParticles::Int = 10, ε::Float64=0.1,
              bandwidth::Float64=-1.0, α::Float64 = 0.9, lik::Bool=true,
              noise::Bool=true, domean::Bool=true, kern::Bool=true,
              trace::Bool=true, hist_tracker::Bool=false, log_interval::Int=10,
              inspect_bw::Bool=false, clip = (-Inf, Inf))

    function calc_dtarget!(gp::GPBase, θ::AbstractVector) #log-target and its gradient
        set_params!(gp, θ; params_kwargs...)
        GaussianProcesses.update_target_and_dtarget!(gp; params_kwargs...)
        pass = true
        if !all(isfinite.(gp.dtarget))
            pass = false
        end
        return pass
    end

    function svgd_kernel(θ::Matrix{Float64}; h::Float64=-1.0, inspect_bw::Bool=false)  # function to calculate the kernel
        pairwise_dist = pairwise(Euclidean(),θ, dims=2).^2  #check that the squared term is correct
        if h<0
            h = median(pairwise_dist)
            h = sqrt(0.5*h/log(size(θ,1)+1))
        end

        #compute the kernel
        # Kxy = Array{Float64}(undef, size(θ, 1), size(θ, 1)) # TODO: This may have to be size(θ, 2)

        Kxy = exp.(-pairwise_dist/(2*h^2))
        dxkxy = -Kxy*θ'
        sumkxy = sum(Kxy; dims=1)
        for i in 1:size(θ, 1)
            dxkxy[:,i] = dxkxy[:,i] + θ[i,:].*vec(sumkxy)
        end
        dxkxy = dxkxy/(h^2)
        if inspect_bw
            return Kxy, dxkxy, h
        else
            return Kxy, dxkxy
        end
    end

    #optimize!(gp)      #find the MAP/MLE as a starting point
    params_kwargs = get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)

    θ_cur = get_params(gp; params_kwargs...)
    D = length(θ_cur)
    θ_particles =   randn(D,nParticles)   #set-up the particles - might be better to sample from the priors
    grad_particles = zeros(D,nParticles)           #set-up the gradients

    if hist_tracker
        particle_tracker = chain(θ_particles, grad_particles)
    end
    if inspect_bw
        h_set = Array{Float64}(undef, nIter, 1)
    end
    fudge_factor = 1e-6   #this is for adagrad
    historical_grad = 0

    for t in ProgressBar(1:nIter)
        for i in 1:nParticles
            while !calc_dtarget!(gp,θ_particles[:,i]) #this is a bit of a hack at the minute and need to think of a better solution. Essentially, if calculating the gradient at one of the particles leads to an instability, then we reset that particle. There's probably a better way of solving this and it's likely and initialisation issue.
                θ_particles[:,i] = randn(D,1)
            end
            grad_particles[:,i] = gp.dtarget
        end
        if inspect_bw
            kxy, dxkxy, h = svgd_kernel(θ_particles; h = bandwidth, inspect_bw=true)
            h_set[t] = h
        else
            kxy, dxkxy = svgd_kernel(θ_particles; h = bandwidth)
        end

        grad_θ = (kxy*grad_particles' + dxkxy) / nParticles
        if clip[1] > -Inf
            if clip[2] < Inf
                grad_θ[grad_θ .< clip[1]] .= clip[1]
                grad_θ[grad_θ .> clip[2]] .= clip[2]
            end
        end

        #adagrad
        if t==1
            historical_grad = historical_grad .+ grad_θ.^2
        else
            historical_grad = α*historical_grad .+ (1-α)*grad_θ.^2
        end

        new_grad = grad_θ#./(fudge_factor .+ sqrt.(historical_grad))
        θ_particles += ε*new_grad'

        if hist_tracker
            if t % log_interval == 0
                push!(particle_tracker, θ_particles)
                push!(particle_tracker, grad_θ'; grads=true)
            end
        end

        # if LinearAlgebra.norm(grad_θ)<10e-6  #early stopping if converged
        #     println("Converged at iteration ", t)
        #     break
        # end
    end
    if hist_tracker
        if inspect_bw
            return θ_particles, particle_tracker, h_set
        else
            return θ_particles, particle_tracker
        end
    else
        return θ_particles
    end
end
