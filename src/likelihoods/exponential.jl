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
    return [Distributions.logpdf(Distributions.Exponential(exp(max(fi,-7e2))),yi) for (fi,yi) in zip(f,y)]
end

function dlog_dens(exponential::Exponential, f::Vector{Float64}, y::Vector{Float64})
    return [Distributions.gradlogpdf(Distributions.Exponential(exp(max(fi,-7e2))),yi) for (fi,yi) in zip(f,y)]
end                   

get_params(exponential::Exponential) = []
num_params(exponential::Exponential) = 0
