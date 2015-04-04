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
