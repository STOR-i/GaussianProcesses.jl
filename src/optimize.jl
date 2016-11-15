@doc """
# Description
A function for optimising the GP hyperparameters based on type II maximum likelihood estimation. This function performs gradient based optimisation using the Optim pacakge to which the user is referred to for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `mean::Bool`: Mean function hyperparameters should be optmized
* `kern::Bool`: Kernel function hyperparameters should be optmized
* `kwargs`: Keyword arguments for the optimize function from the Optim package

# Return:
* `::Optim.MultivariateOptimizationResults{Float64,1}`: optimization results object
""" ->
function optimize!(gp::GP; lik::Bool=false, mean::Bool=true, kern::Bool=true, method::Optim.Optimizer=NelderMead(), kwargs...)
    function ll(hyp::Vector{Float64})
        set_params!(gp, hyp; lik=lik, mean=mean, kern=kern)
        likelihood!(gp)
        return -gp.ll
    end

    init = [gp.v; get_params(gp;  lik=lik, mean=mean, kern=kern)]  # Initial hyperparameter values
    results = optimize(ll, init, method, kwargs...)                     # Run optimizer
    print(results)
end

