# Matern 1/2 isotropic covariance Function

@doc """
# Description
Constructor for the isotropic Matern 1/2 kernel (covariance)

k(x,x') = σ²exp(-d/ℓ), where d=|x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat12Iso <: MaternIso
    ℓ::Float64     # Length scale
    σ2::Float64    # Signal std
    Mat12Iso(ll::Float64, lσ::Float64) = new(exp(ll),exp(2*lσ))
end

function set_params!(mat::Mat12Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 1/2 covariance function only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2.0*hyp[2])
end

get_params(mat::Mat12Iso) = Float64[log(mat.ℓ), log(mat.σ2)/2.0]
get_param_names(mat::Mat12Iso) = [:ll, :lσ]
num_params(mat::Mat12Iso) = 2

metric(mat::Mat12Iso) = Euclidean()
cov(mat::Mat12Iso, r::Float64) = mat.σ2*exp(-r/mat.ℓ)

@inline dk_dll(mat::Mat12Iso, r::Float64) = r/mat.ℓ*cov(mat,r)
