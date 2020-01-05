using Distances, Statistics
"""
    svgd(gp::GPBase; kwargs...)

Runs Stein variational gradient descent to estimate the posterior distribution of the hyperparameters of Gaussian process `GPE` and the latent function in the case of `GPA`.
"""
function svgd(gp::GPBase; nIter::Int=1000, nParticles::Int = 10, ε::Float64=0.1,
              bandwidth::Float64=-1.0, α::Float64 = 0.9, lik::Bool=true,
              noise::Bool=true, domean::Bool=true, kern::Bool=true)
    
    function calc_dtarget!(gp::GPBase, θ::AbstractVector) #log-target and its gradient
        try
            set_params!(gp, θ; params_kwargs...)
            update_target_and_dtarget!(gp; params_kwargs...)
            return true
        catch err
            if !all(isfinite.(θ))
                return false
            elseif isa(err, ArgumentError)
                return false
            elseif isa(err, LinearAlgebra.PosDefException)
                return false
            else
                throw(err)
            end
        end
    end

    function svgd_kernel(θ::Matrix{Float64};h::Float64=-1.0)  # function to calculate the kernel
        pairwise_dist = pairwise(Euclidean(),θ,dims=1)
        if h<0
            h = median(pairwise_dist)
            h = sqrt(0.5*h/log(size(θ,1)+1))
        end
        #compute the kernel
        Kxy = exp.(-pairwise_dist/(2*h^2))
        dxkxy = -Kxy*θ
        sumkxy = sum(Kxy; dims=1)
        for i in 1:size(θ,2)
            dxkxy[:,i] = dxkxy[:,i] + θ[:,i].*vec(sumkxy)
        end
        dxkxy = dxkxy/(h^2)
        return Kxy, dxkxy
    end
    
    optimize!(gp)      #find the MAP/MLE as a starting point
    precomp = init_precompute(gp)
    params_kwargs = get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)

    θ_cur = get_params(gp; params_kwargs...)
    D = length(θ_cur)
    θ_particles =   θ_cur .+ randn(D,nParticles)   #set-up the particles
    grad_particles = zeros(D,nParticles)           #set-up the gradients

    fudge_factor = 1e-6   #this is for adagrad
    historical_grad = 0

    for t in 1:nIter  
        for i in 1:nParticles
            calc_dtarget!(gp, θ_particles[:,i])   #update gradient of log-target 
            grad_particles[:,i] = gp.dtarget      
        end
        kxy, dxkxy = svgd_kernel(θ_particles; h = bandwidth)
        grad_θ = (kxy*grad_particles + dxkxy) / nParticles

        #adagrad
        if t==1
            historical_grad = historical_grad .+ grad_θ.^2
        else
            historical_grad = α*historical_grad .+ (1-α)*grad_θ.^2
        end
        new_grad = grad_θ./(fudge_factor .+ sqrt.(historical_grad))
        θ_particles = θ_particles + ε*new_grad
        
        if LinearAlgebra.norm(grad_θ)<10e-6  #early stopping if converged
            println("Converged at iteration ", t)
            break
        end
    end
    return θ_particles
end


