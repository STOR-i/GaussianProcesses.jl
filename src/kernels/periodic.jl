# Periodic Function 

@doc """
# Description
Constructor for the Periodic kernel (covariance)

k(x,x') = σ²exp(-2sin²(π|x-x'|/p)/ℓ²)
# Arguments:
* `ll::Vector{Float64}`: Log of length scale ℓ
* `lσ::Float64`        : Log of the signal standard deviation σ
* `lp::Float64`        : Log of the period
""" ->
type Periodic <: Isotropic
    ℓ2::Float64
    σ2::Float64
    p::Float64      # Log of period
    Periodic(ll::Float64, lσ::Float64, lp::Float64) = new(exp(2*ll), exp(2*lσ), exp(lp))
end

get_params(pe::Periodic) = Float64[log(pe.ℓ2)/2.0, log(pe.σ2)/2.0, log(pe.p)]
get_param_names(pe::Periodic) = [:ll, :lσ, :lp]
num_params(pe::Periodic) = 3
metric(pe::Periodic) = Euclidean()

function set_params!(pe::Periodic, hyp::Vector{Float64})
    length(hyp) == 3 || throw(ArgumentError("Periodic function has only three parameters"))
    pe.ℓ2, pe.σ2 = exp(2.0*hyp[1:2])
    pe.p = exp(hyp[3])
end

cov(pe::Periodic, r::Float64) = pe.σ2*exp(-2.0/pe.ℓ2*sin(π*r/pe.p)^2)


@inline dk_dll(pe::Periodic, r::Float64) = 4.0*pe.σ2*(sin(π*r/pe.p)^2/pe.ℓ2)*exp(-2.0/pe.ℓ2*sin(π*r/pe.p)^2)  # dK_dlogℓ
@inline dk_dlp(pe::Periodic, r::Float64) = 4.0/pe.ℓ2*pe.σ2*(π*r/pe.p)*sin(π*r/pe.p)*cos(π*r/pe.p)*exp(-2/pe.ℓ2*sin(π*r/pe.p)^2)    # dK_dlogp
@inline function dk_dθp(pe::Periodic, r::Float64, p::Int)
    if p==1
        dk_dll(pe, r)
    elseif p==2
        dk_dlσ(pe, r)
    elseif p==3
        dk_dlp(pe, r)
    else
        return NaN
    end
end
