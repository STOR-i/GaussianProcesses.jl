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
    return Q, V, K
end


function update_Q!(Q, m, V)
    Q.m = m
    Q.V = V
end


function elbo(y::AbstractArray, μ::AbstractArray, Ω::AbstractMatrix, m::AbstractArray, V::AbstractMatrix, ll::Likelihood)
    N = length(y)
    VprodΩ = V * Ω
    return -0.5*(logdet(VprodΩ) - tr(VprodΩ) - transpose(m - μ)*Ω*(m - μ) + N) + var_exp(ll, y, m, V)
end


function push_back!(mat::AbstractArray, idx::Integer)
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
    Q, Ω, K = initialise_Q(gp)
    V = deepcopy(Q.V)
    evaluation = Inf
    
    # Optimise Q
    for i in 1:gp.nobs
        println("Iteration: ", i)
        push_back!(V, i)
        ktilde = K[end, end] - 1/V[end, end]
        v_corner_old = deepcopy(V[end, end])
        for _ in 1:3
            gv = dv_var_exp(gp.lik, gp.y[end], gp.μ[i], V[end, end])
            V[end, end] = 1/(Ω[end, end] - ktilde - 2*gv)
        end
        # Update V 
        # TODO: Check that it should be V[:, end] and not V[end, :]
        V[1, 1] += (V[end, end] - v_corner_old) * (transpose(V[:, end])*V[:, end]) / v_corner_old^2
        V[:, end] = - v_corner_old*V[:, end] / V[end, end]

        # Update K_22
        K[end, end] = ktilde + 1/V[end, end]

        # Update m 
        function m_solver(m::AbstractArray)
            # Return negative solution to enable maximisation
            return  -(maximum(m) - 0.5*transpose(m - gp.μ)*Ω*(m - gp.μ) + var_exp(gp.lik, gp.y, m, V))
        end
        res = optimize(m_solver, Q.m, LBFGS(); autodiff = :forward)
        Q.m = Optim.minimizer(res)
        println(Q.m)
    end
end
