"""
# Description
Constructor for the Student-t likelihood (aka non-standardized Student's t-distribution)

    p(y|ν,f,σ) = Γ((ν+1)/2)/[Γ(ν/2)\sqrt{πν}σ](1+1/ν((y-f)/σ)²)^(-(ν+1)/2),
    where f is the latent function of the GP.

# Arguments:
    * `ν::Int64`:   degrees of freedom
    * `lσ::Float64`: log of scale
# Link:
    * https://en.wikipedia.org/wiki/Student's_t-distribution
"""
type StuTLik <: Likelihood
    ν::Int64    #degrees of freedom
    σ::Float64  #scale
    priors::Array          # Array of priors for likelihood parameters
    StuTLik(ν::Int64,lσ::Float64) = new(ν,exp(lσ),[])
end

#log of probability density
function log_dens(studentT::StuTLik, f::Vector{Float64}, y::Vector{Float64})
    ν = studentT.ν
    σ = studentT.σ
    c = lgamma(0.5*(ν+1)) - lgamma(0.5*ν) - 0.5*log(pi*ν) - log(σ) 
    return [c - (0.5*(ν+1))*log(1+(1/ν)*((yi-fi)/σ)^2) for (fi,yi) in zip(f,y)]
end

#derivative of log pdf wrt latent function
function dlog_dens_df(studentT::StuTLik, f::Vector{Float64}, y::Vector{Float64})
    ν = studentT.ν
    σ = studentT.σ
    return [(ν+1)*(yi-fi)/(ν*σ^2 + (yi-fi)^2) for (fi,yi) in zip(f,y)]
end                   

#derivative of log pdf wrt to parameters
function dlog_dens_dθ(studentT::StuTLik, f::Vector{Float64}, y::Vector{Float64})
    ν = studentT.ν
    σ = studentT.σ
    return σ*[-1/σ+(ν+1)*(yi-fi)^2/(ν*σ^3 + σ*(yi-fi)^2) for (fi,yi) in zip(f,y)]
end                   


function set_params!(studentT::StuTLik, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Student-t likelihood has only one free parameter"))
    studentT.σ = exp(hyp[])
end

get_params(studentT::StuTLik) = Float64[log(studentT.σ)]
num_params(studentT::StuTLik) = 1




