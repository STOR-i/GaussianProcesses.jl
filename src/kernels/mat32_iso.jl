# Matern 3/2 isotropic covariance function

@doc """
# Description
Constructor for the isotropic Matern 3/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/ℓ)exp(-√3*d/ℓ), where d = |x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat32Iso <: MaternIso
    ℓ::Float64             # Length scale 
    σ2::Float64            # Signal std
    priors::Array          # Array of priors for kernel parameters
    Mat32Iso(ll::Float64, lσ::Float64) = new(exp(ll),exp(2*lσ),[])
end

function set_params!(mat::Mat32Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 3/2 only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2.0*hyp[2])
end

get_params(mat::Mat32Iso) = Float64[log(mat.ℓ), log(mat.σ2)/2.0]
get_param_names(mat::Mat32Iso) = [:ll, :lσ]
num_params(mat::Mat32Iso) = 2


metric(mat::Mat32Iso) = Euclidean()
cov(mat::Mat32Iso, r::Float64) = mat.σ2*(1+sqrt(3)*r/mat.ℓ)*exp(-sqrt(3)*r/mat.ℓ)

@inline dk_dll(mat::Mat32Iso, r::Float64) = mat.σ2*(sqrt(3)*r/mat.ℓ)^2*exp(-sqrt(3)*r/mat.ℓ)
