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
    target_cur, grad_cur = -gp.target, -gp.dtarget
    grad = grad_cur
    target = target_cur
    
    for t in 1:nIter
        
        ν_old = randn(D)

        ν = ν_old + 0.5 * ε * grad_cur
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
            post[t,:] = θ_cur
            θ = θ_cur
        end
        
        α = target - 0.5 * ν'ν - target_cur + 0.5 * ν_cur'ν_cur
        u = log(rand())

        if u < α 
            θ_cur = θ
            target_cur = target
            grad_cur = grad
        end
        post[t,:] = θ
    end
    return post
end    

