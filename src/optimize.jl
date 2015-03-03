using Optim

function optimize!(gp::GP)
    function mll(hyp::Vector{Float64})
        set_params!(gp.k, hyp)
        update!(gp)
        return -gp.mLL
    end
    function dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        set_params(gp.k, hyp)
        update!(gp)
        grad[:] = -gp.dmLL
    end
    function mll_and_dmll!(hyp::Vector{Float64}, grad::Vector{Float64})
        set_params!(gp.k, hyp)
        update!(gp)
        grad[:] = -gp.dmLL
        return -gp.mLL
    end

    func = DifferentiableFunction(mll, dmll!, mll_and_dmll!)
    n = num_params(gp.k)
    # Run box minimization (assume lower bound of zero on hyperparameters - must fix this)
    # Could apply a transformation to parameters to use an unconstrained optimization
    l = zeros(n)
    u = fill(Inf, n)
    init = params(gp.k)
    results = fminbox(func, init, l, u)
    print(results)
end
