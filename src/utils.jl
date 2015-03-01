# Returns matrix of distances k where D[i,j] = d(x1[i], x2[j])
#
# Arguments:
#  x1 matrix of observations (each column is an observation)
#  x2 matrix of observations (each column is an observation)
#  d distance function between two vectors
function distance(x1::Matrix{Float64}, x2::Matrix{Float64}, d::Function)
    dim, nobs1 = size(x1)
    nobs2 = size(x2,2)
    dim == size(x2,1) || throw(ArgumentError("Input observation matrices must have consistent dimensions"))
    dist = Array(Float64, nobs1, nobs2)
    for i in 1:nobs1, j in 1:nobs2
        dist[i,j] = d(x1[:,i], x2[:,j])
    end
    return dist
end

# Returns PD matrix of distances D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
#  d distance function between two vectors
function distance(x::Matrix{Float64}, d::Function)
    dim, nobsv = size(x)
    dist = Array(Float64, nobsv, nobsv)
    for i in 1:nobsv
        for j in 1:i
            dist[i,j] = d(x[:,i], x[:,j])
            if i != j; dist[j,i] = dist[i,j]; end;
        end
    end
    return dist
end


# Returns matrix of distances k where D[i,j] = kernel(x1[i], x2[j])
#
# Arguments:
#  x1 matrix of observations (each column is an observation)
#  x2 matrix of observations (each column is an observation)
#  k kernel object
function distance(x1::Matrix{Float64}, x2::Matrix{Float64}, k::Kernel)
    d(x,y) = kern(k, x, y)
    return distance(x1, x2, d)
end

# Returns PD matrix of distances D where D[i,j] = kernel(x1[i], x1[j])
#
# Arguments:
#  x matrix of observations (each column is an observation)
#  k kernel object
function distance(x::Matrix{Float64}, k::Kernel)
    d(x,y) = kern(k, x, y)
    return distance(x, d)
end
