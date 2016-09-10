# Matern 5/2 isotropic covariance function

@doc """
# Description
Constructor for the isotropic Matern 5/2 kernel (covariance)

k(x,x') = σ²(1+√5*d/ℓ + 5d²/3ℓ²)exp(-√5*d/ℓ), where d = |x-x'|
# Arguments:
* `ll::Float64`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat52Iso <: MaternIso
    ℓ::Float64      # Length scale 
    σ2::Float64     # Signal std
    Mat52Iso(ll::Float64, lσ::Float64) = new(exp(ll), exp(2*lσ))
end

function set_params!(mat::Mat52Iso, hyp::Vector{Float64})
    length(hyp) == 2 || throw(ArgumentError("Matern 5/2 only has two parameters"))
    mat.ℓ, mat.σ2 = exp(hyp[1]), exp(2.0*hyp[2])
end
get_params(mat::Mat52Iso) = Float64[log(mat.ℓ), log(mat.σ2)/2.0]
get_param_names(mat::Mat52Iso) = [:ll, :lσ]
num_params(mat::Mat52Iso) = 2

metric(mat::Mat52Iso) = Euclidean()
cov(mat::Mat52Iso, r::Float64) = mat.σ2*(1+sqrt(5)*r/mat.ℓ+5*r^2/(3*mat.ℓ^2))*exp(-sqrt(5)*r/mat.ℓ)

@inline dk_dll(mat::Mat52Iso, r::Float64) = mat.σ2*(5*r^2/mat.ℓ^2)*((1+sqrt(5)*r/mat.ℓ)/3)*exp(-sqrt(5)*r/mat.ℓ)
