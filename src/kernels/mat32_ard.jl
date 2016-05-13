# Matern 3/2 ARD covariance function

@doc """
# Description
Constructor for the ARD Matern 3/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L)exp(-√3*d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat32Ard <: Stationary
    ℓ::Vector{Float64}    # Log of Length scale 
    σ2::Float64            # Log of Signal std
    dim::Int               # Number of hyperparameters
    Mat32Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(ll), exp(2.0*lσ), size(ll,1)+1)
end

function set_params!(mat::Mat32Ard, hyp::Vector{Float64})
    length(hyp) == mat.dim || throw(ArgumentError("Matern 3/2 only has $(mat.dim) parameters"))
    mat.ℓ = exp(hyp[1:(mat.dim-1)])
    mat.σ2 = exp(2.0*hyp[mat.dim])
end

get_params(mat::Mat32Ard) = [log(mat.ℓ); log(mat.σ2)/2.0]
get_param_names(mat::Mat32Ard) = [get_param_names(mat.ℓ, :ll); :lσ]
num_params(mat::Mat32Ard) = mat.dim

metric(mat::Mat32Ard) = WeightedEuclidean(1.0./(mat.ℓ))
kern(mat::Mat32Ard, r::Float64) = mat.σ2*(1+sqrt(3)*r)*exp(-sqrt(3)*r)

function grad_kern(mat::Mat32Ard, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat, x, y)
    exp_r = exp(-sqrt(3)*r)
    wdiff = (x-y)./mat.ℓ
    
    g1 = mat.σ2*(sqrt(3).*wdiff).^2*exp_r
    g2 = 2.0*mat.σ2*(1+sqrt(3)*r)*exp_r
    
    return [g1; g2]
end    
