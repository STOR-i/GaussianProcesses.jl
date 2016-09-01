# Matern 5/2 ARD covariance Function

@doc """
# Description
Constructor for the ARD Matern 5/2 kernel (covariance)

k(x,x') = σ²(1+√3*d/L + 5d²/3L²)exp(-√5*d/L), where d = |x-x'| and L = diag(ℓ₁,ℓ₂,...)
# Arguments:
* `ll::Vector{Float64}`: Log of the length scale ℓ
* `lσ::Float64`: Log of the signal standard deviation σ
""" ->
type Mat52Ard <: StationaryARD
    ℓ2::Vector{Float64}   # Log of Length scale 
    σ2::Float64           # Log of signal std
    Mat52Ard(ll::Vector{Float64}, lσ::Float64) = new(exp(2.0*ll), exp(2.0*lσ))
end

function set_params!(mat::Mat52Ard, hyp::Vector{Float64})
    length(hyp) == num_params(mat) || throw(ArgumentError("Mat52 kernel only has $(num_params(mat)) parameters"))
    d = length(mat.ℓ2)
    mat.ℓ2 = exp(2.0*hyp[1:d])
    mat.σ2 = exp(2.0*hyp[d+1])
end

get_params(mat::Mat52Ard) = [log(mat.ℓ2)/2.0; log(mat.σ2)/2.0]
get_param_names(mat::Mat52Ard) = [get_param_names(mat.ℓ2, :ll); :lσ]
num_params(mat::Mat52Ard) = length(mat.ℓ2) + 1

metric(mat::Mat52Ard) = WeightedEuclidean(1.0./(mat.ℓ2))
cov(mat::Mat52Ard, r::Float64) = mat.σ2*(1+sqrt(5)*r+5/3*r^2)*exp(-sqrt(5)*r)

function grad_kern(mat::Mat52Ard, x::Vector{Float64}, y::Vector{Float64})
    #r = distance(mat,x,y)
    wdiff = (x-y).^2./mat.ℓ2
    r = sqrt(sum(wdiff))
    exp_r = exp(-sqrt(5)*r)
    
    g1 = mat.σ2*(5/3)*(1+sqrt(5)*r)*exp_r.*wdiff # dK_d(log ℓ)
    g2 = 2.0*mat.σ2*(1+sqrt(5)*r+(5/3)*r^2)*exp_r  # dK_d(log σ)
    
    return [g1; g2]
end

function grad_stack!(stack::AbstractArray, mat::Mat52Ard, X::Matrix{Float64}, data::StationaryARDData)
    d = size(X,1)
    R = distance(mat,X)
    exp_R = exp(-sqrt(5)*R)

    stack[:,:,d+1] = 2.0*cov(mat, X)
    part = (5/3) * mat.σ2 .* exp_R .* (1.0 + sqrt(5)*R)
    broadcast!(*, view(stack, :, :, 1:d), (5/3)*mat.σ2, 1 + sqrt(5)*R, exp_R, data.dist_stack, reshape(1.0./mat.ℓ2, (1,1,d)))

    return stack
end
