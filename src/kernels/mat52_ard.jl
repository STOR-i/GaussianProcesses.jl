# Matern 5/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 5/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L + 5d²/3L²)exp(-√5*d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat52Ard <: MaternARD
    iℓ2::Vector{Float64}   # Log of Length scale 
    σ2::Float64           # Log of signal std
    Mat52Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(-2.0*ll), exp(2.0*lσ))
end

function set_params!(mat::Mat52Ard, hyp::Vector{Float64})
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat52 kernel only has $(num_params(mat)) parameters"))
    d = length(mat.iℓ2)
    mat.iℓ2 = exp(-2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat52Ard) = [-log(mat.iℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat52Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat52Ard) = length(mat.iℓ2) + 1

metric(mat::Mat52Ard) = WeightedEuclidean(mat.iℓ2)
cov(mat::Mat52Ard, r::Float64) = mat.σ2*(1+sqrt(5)*r+5/3*r^2)*exp(-sqrt(5)*r)

dk_dll(mat::Mat52Ard, r::Float64, wdiffp::Float64) = mat.σ2*(5/3)*(1+sqrt(5)*r)*exp(-sqrt(5)*r).*wdiffp
