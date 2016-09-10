# Matern 1/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 1/2 kernel (covariance)

k(x,x') = σ²exp(-d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat12Ard <: MaternARD
    iℓ2::Vector{Float64}     # Inverse squared Length scale
    σ2::Float64              # Log of signal std
    Mat12Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(-2.0*ll),exp(2.0*lσ))
end

function set_params!(mat::Mat12Ard, hyp::Vector{Float64})
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat12 kernel only has $(num_params(mat)) parameters"))
    d = length(mat.iℓ2)
    mat.iℓ2  = exp(-2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat12Ard) = [-log(mat.iℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat12Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat12Ard) = length(mat.iℓ2) + 1

metric(mat::Mat12Ard) = WeightedEuclidean(mat.iℓ2)
cov(mat::Mat12Ard, r::Float64) = mat.σ2*exp(-r)

dk_dll(mat::Mat12Ard, r::Float64, wdiffp::Float64) = wdiffp/r*cov(mat,r)
