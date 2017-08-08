@doc """
    # Description
    A function for deriving a variational approximation, q(θ|φ), to the posterior distribution, p(θ|X,y) which minimises the Kullback-Leibler divergence between p(·|X,y) and  q(·|φ), where θ are model parameters, which in the case of non-Gaussian likelihoods, includes the latent function, f.
        
Minimising the KL divergence is done using the Optim.jl, which the user is referred to for further details.

    # Arguments:
    * `gp::GPBase`: Predefined Gaussian process type
    * `nSamps::Int64`: Number of Monte Carlo samples with 100 set as default
    * `kwargs`: Keyword arguments for the optimize function from the Optim package

    # Return:
    * `samples::Matrix{Float64}`: samples from the variational approximation
    """ 
function vi(gp::GPBase; method=LBFGS(), kwargs...)
    func = get_optim_target(gp)
    init = zeros(2*length(get_params(gp)))  # Initial hyperparameter values
    results = optimize(func,init; method=method, kwargs...)      # Run optimizer
    return results
end

function get_optim_target(gp::GPBase)
    
    function ltarget(params::Vector{Float64})
        try
            nSamps = 100
            K = div(length(params),2)
            target = Float64[]
            eta = randn(K,nSamps)
            samples = params[1:K] .+ diagm(exp.(params[K+1:end]))*eta
            for s in 1:nSamps
                set_params!(gp, samples[:,s])
                append!(target,update_target!(gp))
            end
            est_target = mean(target)
            return -(est_target +0.5*K*(1+log(2*pi)) +sum(params[K+1:end]))
        catch err
            if !all(isfinite.(params))
                println(err)
                return Inf
            elseif !isfinite(est_target)
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end        
    end

    function ltarget_and_dltarget!(params::Vector{Float64}, grad::Vector{Float64})
        try
            nSamps = 100
            K = div(length(params),2)
            dtarget = Array{Float64}(K,nSamps)
            eta = randn(K,nSamps)
            samples = params[1:K] .+ diagm(exp.(params[K+1:end]))*eta
            for s in 1:nSamps
                set_params!(gp, samples[:,s])
                dtarget[:,s] = update_target_and_dtarget!(gp)
            end
            dtarget = vcat(dtarget,dtarget.*eta.*exp.(params[K+1:end]))
            est_dtarget = mean(dtarget,2)
            est_dtarget[K+1:end] += 1.0 
            grad[:] = -est_dtarget
            return grad
        catch err
            if !all(isfinite.(params))
                println(err)
                return Inf
            elseif !all(isfinite.(grad))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                return Inf
            else
                throw(err)
            end
        end 
    end
    
    function dltarget!(params::Vector{Float64}, grad::Vector{Float64})
        ltarget_and_dltarget!(params::Vector{Float64}, grad::Vector{Float64})
    end

    func = OnceDifferentiable(ltarget, dltarget!, ltarget_and_dltarget!)
    return func
end
