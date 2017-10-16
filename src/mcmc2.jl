@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Klara package and the user is referred to this package for further details.

# Arguments:
    * `gp::GP`: Predefined Gaussian process type
    * `sampler::Klara.MCSampler`: MCMC sampler selected from the Klara package
    * `nIter::Int`: Number of MCMC iterations
    * `burnin::Int`: Number of burn-in MCMC iterations (i.e. number of initial samples discarded)
    * `thin::Int`: Thin factor
""" ->
function mcmc(gp::GPBase;
              sampler::Klara.MCSampler=Klara.MALA(0.1),
              nIter::Int = 1000,
              burnin::Int = 0,
              thin::Int = 1)
    
    function logpost(hyp::Vector{Float64})  #log-target
        set_params!(gp, hyp)
        return update_target!(gp)
    end

    function dlogpost(hyp::Vector{Float64}) #gradient of the log-target
        Kgrad_buffer = Array{Float64}(gp.nobsv, gp.nobsv)
        set_params!(gp, hyp)
        update_target_and_dtarget!(gp)
        return gp.dtarget
    end
    
    start = get_params(gp)
    starting = Dict(:p=>start)
    q = BasicContMuvParameter(:p, logtarget=logpost, gradlogtarget=dlogpost) 
    model = likelihood_model(q, false)                               #set-up the model
    job = BasicMCJob(model, sampler, BasicMCRange(nsteps=nIter, thinning=thin, burnin=burnin), starting)   #set-up MCMC job
    print(job)                                             #display MCMC set-up for the user
    
    run(job)                          #Run MCMC
    chain = Klara.output(job)         # Extract chain
    set_params!(gp,start)      #reset the parameters stored in the GP to original values
    return chain.value
end    

