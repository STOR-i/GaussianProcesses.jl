function optimise!(gp::GPR, x::AbstractArray, y::AbstractArray)
    θ = Flux.Params(return_params(gp.Kernel))
    loss(x, y)
    # TODO: Write MLL function that returns a function to evaluate the MLL using just x and y
