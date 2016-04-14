# Matern 5/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 5/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L + 5d²/3L²)exp(-√5*d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat52Ard <: Stationary
    ℓ::Vector{Float64}   # Log of Length scale 
    σ2::Float64           # Log of signal std
    dim::Int              # Number of hyperparameters
    Mat52Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(ll), exp(2.0*lσ), size(ll,1)+1)
end

function set_params!(mat::Mat52Ard, hyp::Vector{Float64})
    length(hyp) == mat.dim || throw(ArgumentError("Matern 5/2 only has $(mat.dim) parameters"))
    mat.ℓ = exp(hyp[1:(mat.dim-1)])
    mat.σ2 = exp(2.0*hyp[mat.dim])
end

get_params(mat::Mat52Ard) = [log(mat.ℓ); log(mat.σ2)/2.0]
get_param_names(mat::Mat52Ard) = [get_param_names(mat.ℓ, :ll); :lσ]
num_params(mat::Mat52Ard) = mat.dim

metric(mat::Mat52Ard) = WeightedEuclidean(1.0./(mat.ℓ))
kern(mat::Mat52Ard, r::Float64) = mat.σ2*(1+sqrt(5)*r+5/3*r^2)*exp(-sqrt(5)*r)


function grad_kern(mat::Mat52Ard, x::Vector{Float64}, y::Vector{Float64})
    r = distance(mat,x,y)
    exp_r = exp(-sqrt(5)*r)
    wdiff = (x-y)./mat.ℓ
    
    g1 = mat.σ2.*((5*wdiff.^2).*(1+sqrt(5).*wdiff)/3).*exp_r #dK_d(log ℓ)
    g2 = 2.0*mat.σ2*(1+sqrt(5)*r+5/3*r^2)*exp_r              #dK_d(log σ)
    
    return [g1; g2]
end
