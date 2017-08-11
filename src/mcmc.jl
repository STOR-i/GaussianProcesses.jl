@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Klara package and the user is referred to this package for further details.

    # Arguments:
    * `gp::GP`: Predefined Gaussian process type
        * `nIter::Int`: Number of MCMC iterations
        * `ε::Real`: Stepsize parameter
        * `L::Int`: Number of leapfrog steps
        """ ->
function mcmc(gp::GPBase; nIter::Int=1000, ε::Float64=0.05, L::Int=5)

    θ = get_params(gp)
    D = length(θ)
    post = Array{Float64}(nIter,D)     #posterior samples
    post[1,:] = θ
    
    update_target_and_dtarget!(gp)
    target, grad = -gp.target, -gp.dtarget

    for t in 1:nIter
        θ_old, target_old, grad_old = θ, copy(gp.target), copy(grad)
        
        ν_old = randn(D)

        ν = ν_old + 0.5 * ε * grad
        reject = false
        for l in 1:L
            θ += ε * ν
            set_params!(gp,θ)
            try
                update_target_and_dtarget!(gp)
            catch
                reject =true
                break
            end
            target, grad = -gp.target, -gp.dtarget
            ν += ε * grad
        end
        ν -= 0.5*ε * grad

        if reject
            post[t,:] = θ_old
            θ = θ_old
        end
        
        α = target - 0.5 * ν'ν - target_old + 0.5 * ν_old'ν_old
        u = log(rand())

        if u < α 
            post[t,:] = θ

        else  
            post[t,:] = θ_old
            θ, target, grad = θ_old, target_old, grad_old
        end
    end
    return post
end    

