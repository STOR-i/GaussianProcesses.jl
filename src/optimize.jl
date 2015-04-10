@doc """
# Description
A function for optimising the GP hyperparameters based on type II maximum likelihood estimation. This function performs gradient based optimisation using the Optim pacakge to which the user is referred to for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `kwargs`: Keyword arguments for the optimize function from the Optim package
""" ->
function optimize!(gp::GP; kwargs...)
    function mll(hyp::Vector{Float64})
        set_params!(gp, hyp)
        update!(gp)
        return -gp.mLL
    end
    function dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        set_params!(gp, hyp)
        update!(gp)
        grad[:] = -gp.dmLL
    end
    function mll_and_dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        set_params!(gp, hyp)
        update!(gp)
        grad[:] = -gp.dmLL
        return -gp.mLL
    end

    func = DifferentiableFunction(mll, dmll!, mll_and_dmll!)
    init = get_params(gp)                      # Initial hyperparameter values
    results=optimize(func,init; kwargs...) # Run optimizer
    print(results)
end
