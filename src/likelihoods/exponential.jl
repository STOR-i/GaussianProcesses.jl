"""
        # Description
        Constructor for the Exponential likelihood


        # Arguments:
        * `θ::Float64`: rate parameter is the exponential of the latent function, i.e. θ = exp(f)
        """
type ExpLik <: Likelihood
    ExpLik() = new()
end

#log of probability density
function log_dens(exponential::ExpLik, f::Vector{Float64}, y::Vector{Float64})
    #where we exponentiate for positivity f = exp(fi) 
    return [-fi - exp(-fi)*yi for (fi,yi) in zip(f,y)]
end

#derivative of pdf wrt latent function
function dlog_dens_df(exponential::ExpLik, f::Vector{Float64}, y::Vector{Float64})
    return [(yi*exp(-fi)-1) for (fi,yi) in zip(f,y)]
end                   

get_params(exponential::ExpLik) = []
num_params(exponential::ExpLik) = 0




