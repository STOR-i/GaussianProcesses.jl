@doc """
# Description
A function for optimising the GP hyperparameters based on type II maximum likelihood estimation. This function performs gradient based optimisation using the Optim pacakge to which the user is referred to for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `kwargs`: Keyword arguments for the optimize function from the Optim package
""" ->
function optimize!(gp::GP; noise::Bool=true, mean::Bool=true, kern::Bool=true, kwargs...)
    println("noise=$(noise)")
    println("mean=$(mean)")
    println("kern=$(kern)")
    function mll(hyp::Vector{Float64})
        set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
        update_mll!(gp)
        return -gp.mLL
    end
    function dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
        update_mll_and_dmll!(gp; noise=noise, mean=mean, kern=kern)
        grad[:] = -gp.dmLL
    end
    function mll_and_dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        set_params!(gp, hyp; noise=noise, mean=mean, kern=kern)
        update_mll_and_dmll!(gp; noise=noise, mean=mean, kern=kern)
        grad[:] = -gp.dmLL
        return -gp.mLL
    end

    func = DifferentiableFunction(mll, dmll!, mll_and_dmll!)
    init = get_params(gp;  noise=noise, mean=mean, kern=kern)  # Initial hyperparameter values
    results=optimize(func,init; kwargs...)                     # Run optimizer
    print(results)
end
