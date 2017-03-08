@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Klara package and the user is referred to this package for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `start::Vector{Float64}`: Select a starting value, default is taken as current GP parameters
* `sampler::Klara.MCSampler`: MCMC sampler selected from the Klara package
* `mcrange::Klara.BasicMCRange`: Choose number of MCMC iterations and burnin length, default is nsteps=5000, burnin = 1000
""" ->
function mcmc(gp::GPBase;
              start::Vector{Float64}=get_params(gp),
              sampler::Klara.MCSampler=Klara.MALA(0.1),
              mcrange::Klara.BasicMCRange=Klara.BasicMCRange(nsteps=5000, burnin=1000))

    
    
    function logtarget(hyp::Vector{Float64})  #log-target
        try
            set_params!(gp, hyp)
            return update_target!(gp)
        catch err
            if !all(isfinite(hyp))
                println(err)
                return -Inf
            elseif isa(err, ArgumentError)
                println(err)
                return -Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return -Inf
            else
                throw(err)
            end
        end        
    end

    function dlogtarget(hyp::Vector{Float64}) #gradient of the log-target
        set_params!(gp, hyp)
        return update_target_and_dtarget!(gp)
    end
    
    starting = Dict(:p=>start)
    q = BasicContMuvParameter(:p, logtarget=logtarget, gradlogtarget=dlogtarget) 
    model = likelihood_model(q, false)               #set-up the model
    tune = AcceptanceRateMCTuner(0.6, verbose=true)  #set length of tuning (default to burnin length)
    job = BasicMCJob(model, sampler, mcrange, starting)   #set-up MCMC job
    print(job)                                #display MCMC set-up for the user
    run(job)
    chain = Klara.output(job)
    set_params!(gp,start)      #reset the parameters stored in the GP to original values
    return chain.value
end    
    
