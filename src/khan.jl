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
    @assert length(μ) == length(m)
    @assert size(V) == size(Ω)
    println("y: ", size(y))
    println("μ: ", size(μ))
    println("Ω: ", size(Ω))
    println("m: ", size(m))
    println("V: ", size(V))
    N = length(y)
    VprodΩ = V * Ω
    vexp =  var_exp(ll, y, m, V)
    return 0.5*(logdet(VprodΩ) - tr(VprodΩ) - transpose(m - μ)*Ω*(m - μ) + N) + vexp
end


function push_back(mat::AbstractMatrix, idx::Integer)
    # TODO: Possibly more efficient way to do this.
    ori_size = size(mat)
    target = mat[:, idx]
    temp =  mat[:,setdiff(1:end, idx)]
    mat = cat(temp, target, dims=2)
    return mat
end


function push_back(mat::AbstractVector)
    target = mat[1]
    append!(mat[2:end], target)
    return mat
end


function vi(gp::GPBase; nits::Int64=100)
    # Initialise log-target and log-target's derivative
    mcmc(gp; nIter=1)

    # Initialise variational approximation
    Q, Ω, K = initialise_Q(gp)
    V = deepcopy(Q.V)
    evaluation = Inf
    y = deepcopy(gp.y)
    μ = deepcopy(gp.μ)

    # Optimise Q
    for j in 1:nits
        for i in 1:gp.nobs
            preV = deepcopy(V)
            V = push_back(V, 1)
            @assert V != preV
            Ω = push_back(Ω, 1)
            K = push_back(K, 1)
            y = push_back(y)
            μ = push_back(μ)

            ktilde = K[end, end] - 1/V[end, end]
            v_corner_old = deepcopy(V[end, end])

            for _ in 1:10
                gv = dv_var_exp(gp.lik, y[end], μ[end], V[end, end])
                V[end, end] = 1/(Ω[end, end] - ktilde - 2*gv)
                println("Gradient: ", gv)
                println("Derivative: ", V[end, end])
            end
            println("----------------------------") 
            # Update V 
            V[1:(end-1), 1:(end-1)] += (V[end, end] - v_corner_old) * (V[1:(end-1), end] * transpose(V[1:(end-1), end])) / v_corner_old^2
            V[1:(end-1), end] = -v_corner_old*V[1:(end-1), end] / V[end, end]
            V[end, 1:(end-1)] = V[1:(end-1), end]
            
            # Update K_22
            K[end, end] = ktilde + 1/V[end, end]
        end
        update_Q!(Q, Q.m, V)
        println("V: ", V)    
        # Update m 
            function m_solver(m::AbstractArray)
                # Return negative solution to enable maximisation
                return  -(-0.5*transpose(m - gp.μ)*Ω*(m - gp.μ) + var_exp(gp.lik, gp.y, m, V))
            end
            res = optimize(m_solver, Q.m, Newton())# LBFGS(); autodiff = :forward)
            Q.m = Optim.minimizer(res)
        end
    return Q
end
