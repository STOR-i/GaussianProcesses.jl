@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Klara package and the user is referred to this package for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `start::Vector{Float64}`: Select a starting value, default is taken as current GP parameters
* `sampler::Klara.MCSampler`: MCMC sampler selected from the Klara package
* `mcrange::Klara.BasicMCRange`: Choose number of MCMC iterations and burnin length, default is nsteps=5000, burnin = 1000
""" ->
function mcmc(gp::GPMC;
              start::Vector{Float64}=get_params(gp),
              sampler::Klara.MCSampler=Klara.MALA(0.1),
              mcrange::Klara.BasicMCRange=Klara.BasicMCRange(nsteps=5000, burnin=1000))

    
    # prior=Array(Distribution,npara)
    # if eltype(prior)==Distributions.Distribution
    # logprior(prior,hyp::Vector{Float64}) = sum([logpdf(prior[i],hyp[i]) for i=1:npara])
    # gradlogprior(prior,hyp::Vector{Float64}) = sum([gradlogpdf(prior[i],hyp[i]) for i=1:npara])
    # else
    #  logprior(prior,hyp::Vector{Float64}) = logpdf(prior,hyp)
    #  gradlogprior(prior,hyp::Vector{Float64}) = sum(gradlogpdf(prior,hyp))
    # end

    
    function logpost(hyp::Vector{Float64})  #log-target
        set_params!(gp, hyp)
        return log_posterior(gp)
    end

    function dlogpost(hyp::Vector{Float64}) #gradient of the log-target
        Kgrad_buffer = Array(Float64, gp.nobsv, gp.nobsv)
        set_params!(gp, hyp)
        return dlog_posterior(gp, Kgrad_buffer)
    end
    
    starting = Dict(:p=>start)
    q = BasicContMuvParameter(:p, logtarget=logpost, gradlogtarget=dlogpost) 
    model = likelihood_model(q, false)                               #set-up the model
    tune = AcceptanceRateMCTuner(0.6, verbose=true)                     #set length of tuning (default to burnin length)
    job = BasicMCJob(model, sampler, mcrange, starting,tuner=tune)   #set-up MCMC job
    print(job)                                                       #display MCMC set-up for the user
    run(job)
    chain = Klara.output(job)
    set_params!(gp,start)      #reset the parameters stored in the GP to original values
    return chain.value
end    
    
