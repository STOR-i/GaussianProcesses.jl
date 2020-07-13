function get_params(model::GPR)
    μ_params  = gp.MeanFunc
    Σ_params = gp.Kernel
end
