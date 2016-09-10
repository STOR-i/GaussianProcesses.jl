# Matern 3/2 ARD covariance function

@doc """
# Description
Constructor for the ARD Matern 3/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L^2)exp(-√3*d/L^2), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat32Ard <: MaternARD
    iℓ2::Vector{Float64}     # Inverse squared length scale
    σ2::Float64              # Signal variance
    Mat32Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(-2.0*ll), exp(2.0*lσ))
end

function set_params!(mat::Mat32Ard, hyp::Vector{Float64})
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat32 kernel only has $(num_params(mat)) parameters"))
    d=length(mat.iℓ2)
    mat.iℓ2 = exp(-2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat32Ard) = [-log(mat.iℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat32Ard) = [get_param_names(mat.iℓ2, :ll); :lσ]
num_params(mat::Mat32Ard) = length(mat.iℓ2) + 1

metric(mat::Mat32Ard) = WeightedEuclidean(mat.iℓ2)
cov(mat::Mat32Ard, r::Float64) = mat.σ2*(1+sqrt(3)*r)*exp(-sqrt(3)*r)

dk_dll(mat::Mat32Ard, r::Float64, wdiffp::Float64) = 3.0*mat.σ2*wdiffp*exp(-sqrt(3)*r)
