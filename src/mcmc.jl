@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Klara package and the user is referred to this package for further details.

    # Arguments:
    * `gp::GP`: Predefined Gaussian process type
        * `nIter::Int`: Number of MCMC iterations
        * `ε::Real`: Stepsize parameter
        * `L::Int`: Number of leapfrog steps
        """ ->
function mcmc(gp::GPBase; nIter::Int=1000, ε::Real=0.01, Lmin::Int=5, Lmax::Int=15)

    function calc_target(gp::GPBase,θ::Vector{Float64}) #log-target and its gradient 
        try
            set_params!(gp, θ)
            update_target_and_dtarget!(gp)
            return true
        catch err
            if !all(isfinite.(θ))
                return false
            elseif isa(err, ArgumentError)
                return false
            elseif isa(err, Base.LinAlg.PosDefException)
                return false
            else
                throw(err)
            end
        end        
    end


    θ_cur = get_params(gp)
    D = length(θ_cur)
    post = Array{Float64}(nIter,D)     #posterior samples
    post[1,:] = θ_cur
    
    update_target_and_dtarget!(gp)
    target_cur, grad_cur = -gp.target, -gp.dtarget
    
    for t in 1:nIter
        θ, target, grad = θ_cur, target_cur, grad_cur
        
        ν_cur = randn(D)        
        ν = ν_cur + 0.5 * ε * grad
        
        reject = false
        for l in 1:rand(Lmin:Lmax)
            θ += ε * ν
            if  !calc_target(gp,θ)
                reject=true
                break
            end
            target, grad = -gp.target, -gp.dtarget
            ν += ε * grad
        end
        ν -= 0.5*ε * grad

        if reject
            post[t,:] = θ_cur
        else        
            α = -target - 0.5 * ν'ν + target_cur + 0.5 * ν_cur'ν_cur
            u = log(rand())

            if u < α 
                θ_cur = θ
                target_cur = target
                grad_cur = grad
            end
            post[t,:] = θ_cur
        end
    end
    return post
end    


