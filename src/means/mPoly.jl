# Polynomial mean function

@doc """
# Description
Constructor for the Polynomial mean function

m(x) = Σ_d xᵈβᵀ_d
# Arguments:
* `β::Matrix{Float64}`: A Dxd matrix of coefficients where D is the dimension of xᵈ and d is the degree of the polynomial
""" ->
type MeanPoly <: Mean
    β::Matrix{Float64}    # Polynomial coefficients
    dim::Int              # Dimension
    deg::Int              # Polynomial degree
    MeanPoly(β::Matrix{Float64}) = new(β,size(β, 1), size(β, 2))
end

function mean(mPoly::MeanPoly,x::Matrix{Float64})
    dim, nobsv = size(x)
    dim == mPoly.dim || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    z = zeros(nobsv)
    for i in 1:nobsv
        for j in 1:mPoly.deg
            z[i] += dot(x[:, i].^j, mPoly.β[:,j])
        end    
    end
    return z
end

get_params(mPoly::MeanPoly) = vec(mPoly.β)
get_param_names(mPoly::MeanPoly) = get_param_names(mPoly.β, :β)
num_params(mPoly::MeanPoly) = mPoly.dim * mPoly.deg
function set_params!(mPoly::MeanPoly, hyp::Vector{Float64})
    num_param = mPoly.dim * mPoly.deg
    length(hyp) == num_param || throw(ArgumentError("Polynomial mean function has $(num_param) parameters"))
    mPoly.β = reshape(hyp,mPoly.dim,mPoly.deg)
end


function grad_mean(mPoly::MeanPoly, x::Vector{Float64})
    dM_theta = Array(Float64,mPoly.dim,mPoly.deg)
    
    for i in 1:mPoly.dim
        for j in 1:mPoly.deg
            dM_theta[i,j] = x[i].^j
        end
    end
    return vec(dM_theta)
end
