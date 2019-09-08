mutable struct Approx
    m::AbstractArray
    V::AbstractMatrix # TODO: Rewrite as PDMat
end


function initialise_Q(gp::GPBase)
    # TODO: Use PDMats for the below
    V = cov(gp.kernel, gp.x, gp.data)
    Ω = inv(V)
    K = deepcopy(Ω)
    m = mean(gp.mean, gp.x)
    Q = Approx(m, V)
    return Q
end


function push_back!(mat::AbstractArray, idx::Integer)
    # TODO: Make bang function
    # TODO: Possibly more efficient way to do this.
    ori_size = size(mat)
    target = mat[:, idx]
    temp =  mat[:,setdiff(1:end, idx)]
    mat = cat(temp, target, dims=2)
    @assert size(mat) == ori_size
end


function vi(gp::GPBase; nits::Int64=1000)
    # Initialise log-target and log-target's derivative
    mcmc(gp; nIter=1)

    # Initialise variational approximation
    Q = initialise_Q(gp)
    V = deepcopy(Q.V)
    evaluation = Inf

    # Optimise Q
    for i in 1:5 # TODO: Switch 5 to gp.nobs
        push_back!(V, i)
        ktilde = K[end, end] - 1/V[end, end]
        v_corner_old = deepcopy(V[end, end])
        for _ in 1:3
            V[end, end] = 1/(Ω[end, end] - ktile)# - 2g)
        end
    end
end
