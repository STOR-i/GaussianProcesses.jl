type GaussianProcess
    mean::Function
    cov::Function
    observations::Dict{Vector{Float64}, Float64}
end

