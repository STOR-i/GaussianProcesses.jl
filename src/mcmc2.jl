@doc """
    # Description
    A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Mambda package and the user is referred to this package for further details.

    # Arguments:
    * `gp::GP`: Predefined Gaussian process type
    * `init::Vector{Float64}`: Select a starting value, default is taken as current GP parameters
    * `sampler::Klara.MCSampler`: MCMC sampler selected from the Klara package
    * `mcrange::Klara.BasicMCRange`: Choose number of MCMC iterations and burnin length, default is nsteps=5000, burnin = 1000
    """ ->
function mcmc(gp::GPMC,
              nIters::Int64,
              burnin::Int64,
              epsilon::Float64,
              L::Int64;
              init::Vector{Float64}=get_params(gp))

    ## Log-transformed Posterior and Gradient Vector
    logfgrad = function(hyp::DenseVector)
        set_params!(gp, hyp)
        update_target_and_dtarget!(gp)
        return gp.lp, gp.dlp
    end
        
   #     sim = Chains(nIters, length(get_params(gp)), start = (burnin + 1))
        out = Array(Float64,nIters, length(get_params(gp)))
        theta = HMCVariate(init, epsilon, L, logfgrad)
        for i in 1:nIters
            sample!(theta)
            out[i, :] = theta;
        end
        set_params!(gp,init)      #reset the parameters stored in the GP to their original values
        return out[(burnin+1):end,:]
    end    
    


