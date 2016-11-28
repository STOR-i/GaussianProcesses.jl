"""
        # Description
        Constructor for the Exponential likelihood


        # Arguments:
        * `θ::Float64`: rate parameter
        """
type Exponential <: Likelihood
    Exponential() = new()
end

function log_dens(exponential::Exponential, f::Vector{Float64}, y::Vector{Float64})
    #where f = exp(fi), check for zero
    return [fi - exp(fi)*yi for (fi,yi) in zip(f,y)]
end

function dlog_dens_df(exponential::Exponential, f::Vector{Float64}, y::Vector{Float64})
    return [(1/exp(fi) - yi)*exp(fi) for (fi,yi) in zip(f,y)]
end                   

get_params(exponential::Exponential) = []
num_params(exponential::Exponential) = 0




