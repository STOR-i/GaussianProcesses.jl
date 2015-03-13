# Polynomial mean function

type mPOLY <: Mean
    β::Matrix{Float64}    # Polynomial coefficients
    dim::Int              # Dimension
    deg::Int              # Polynomial degree
    mPOLY(β::Matrix{Float64}) = new(β,size(β, 1), size(β, 2))
end

function meanf(mPoly::mPOLY,x::Matrix{Float64})
    dim, nobsv = size(x)
    dim == mPoly.dim || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    z = zeros(nobsv)
    for i in 1:nobsv
        for j in 1:mPoly.deg
            z[i] += dot(x[:, i].^j, mPoly.β[:j])
        end    
    end
    return z
end

params(mPoly::mPOLY) = vec(mPoly.β)
num_params(mPoly::mPOLY) = mPoly.dim * mPoly.deg
function set_params!(mPoly::mPOLY, hyp::Vector{Float64})
    num_param = mPoly.dim * mPoly.deg
    length(hyp) == num_param || throw(ArgumentError("Polynomial mean function has $(num_param) parameters"))
    mPoly.β = hyp
end

# Needs fixing...
function grad_meanf(mPoly::mPOLY, x::Vector{Float64})
    nobsv = size(x,2)
    z = zeros(nobsv,1)
    for i in 1:mPoly.d
     z = z + x'.^i
    end
    dM_theta = z
    return dM_theta
end
