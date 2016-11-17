import Base.show
# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.

# Constructors:
    GP(X, y, m, k, logNoise)
    GPMC(; m=MeanZero(), k=SE(0.0, 0.0), logNoise=-1e8) # observation-free constructor

# Arguments:
* `X::Matrix{Float64}`: Input observations
* `y::Vector{Float64}`: Output observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `logNoise::Float64` : Log of the standard deviation for the observation noise. The default is -1e8, which is equivalent to assuming no observation noise.

# Returns:
* `gp::GPMC`            : Gaussian process object, fitted to the training data if provided
""" ->
type GPMC{T<:Real}
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    lik::Likelihood         # Likelihood is Gaussian for GPMC regression
    # logNoise::Float64       # log standard deviation of observation noise
    
    # Observation data
    nobsv::Int              # Number of observations
    X::Matrix{Float64}      # Input observations
    y::Vector{T}            # Output observations
    v::Vector{Float64}      # Vector of latent (whitened) variables - N(0,1)
    data::KernelData        # Auxiliary observation data (to speed up calculations)
    dim::Int                # Dimension of inputs
    
    # Auxiliary data
    μ::Vector{Float64} 
    Σ::Matrix{Float64} 
    cK::AbstractPDMat       # (k + exp(2*obsNoise))
    ll::Float64             # Log-likelihood of general GPMC model
    

    function GPMC{S<:Real}(X::Matrix{Float64}, y::Vector{S}, m::Mean, k::Kernel, lik::Likelihood)
        dim, nobsv = size(X)
        v = zeros(nobsv)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        #=
        This is a vanilla implementation of a GPMC with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, where
        v ~ N(0, I)
        f = Lv + m(x)
        with
        L L^T = K
        =#
        gp = new(m, k, lik, nobsv, X, y, v, KernelData(k, X), dim)
        likelihood!(gp)
        return gp
    end
end

# Creates GP object for 1D case
GPMC(x::Vector{Float64}, y::Vector, meanf::Mean, kernel::Kernel, lik::Likelihood) = GPMC(x', y, meanf, kernel, lik)



#Likelihood function of general GP model
function likelihood!(gp::GPMC)
    # log p(Y|v,θ) 
    gp.μ = mean(gp.m,gp.X)
    gp.Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(gp.Σ + 1e-8*eye(gp.nobsv))
    F = gp.cK*gp.v + gp.μ
    gp.ll = sum(log_dens(gp.lik,F,gp.y))
end

function d_likelihood!(gp::GPMC;
                       lik::Bool=true,  # include gradient component for the likelihood parameters
                       mean::Bool=true, # include gradient components for the mean parameters
                       kern::Bool=true, # include gradient components for the spatial kernel parameters
)
    # dlog p(Y|v,θ)
    n_lik_params = num_params(gp.lik)
    n_mean_params = num_params(gp.m)
    n_kern_params = num_params(gp.k)

    likelihood!(gp)
    gp.dll = Array(Float64,gp.nobsv + lik*n_lik_params + mean*n_mean_params + kern*n_kern_params)

    gp.dll[1:gp.nobsv] = dlog_dens(gp.lik, gp.cK*gp.v + gp.μ, gp.y)

    i=1  #NEEDS COMPLETING
    if lik
        Mgrads = grad_stack(gp.m, gp.X)
        for j in 1:n_lik_params
            gp.dll[i] = dot(Mgrads[:,j],gp.alpha)
            i += 1
        end
    end
    if mean
        Mgrads = grad_stack(gp.m, gp.X)
        for j in 1:n_mean_params
            gp.dll[i] = dot(Mgrads[:,j],gp.alpha)
            i += 1
        end
    end
    if kern
        for iparam in 1:n_kern_params
            grad_slice!(Kgrad, gp.k, gp.X, gp.data, iparam)
            gp.dll[i] = vecdot(Kgrad,ααinvcKI)/2.0
            i+=1
        end
    end
end    


function conditional(gp::GPMC, X::Matrix{Float64})
    n = size(X, 2)
    Σ = cov(gp.k, gp.X, gp.data)
    gp.cK = PDMat(Σ + 1e-6*eye(gp.nobsv))
    cK = cov(gp.k, X, gp.X)
    Lck = whiten(gp.cK, cK')
    Sigma_raw = cov(gp.k, X) - Lck'Lck # Predictive covariance
    # Hack to get stable covariance
    fSigma = try PDMat(Sigma_raw) catch; PDMat(Sigma_raw+1e-8*sum(diag(Sigma_raw))/n*eye(n)) end
    fmu = mean(gp.m,X) + Lck*gp.v        # Predictive mean
    return fmu, fSigma
end

@doc """
    # Description
    Calculates the posterior mean and variance of Gaussian Process at specified points

    # Arguments:
    * `gp::GP`: Gaussian Process object
    * `X::Matrix{Float64}`:  matrix of points for which one would would like to predict the value of the process.
                           (each column of the matrix is a point)

    # Returns:
    * `(mu, Sigma)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
                                                        process at the specified points
    """ ->
function predict(gp::GPMC, X::Matrix{Float64}; full_cov::Bool=false)
    size(X,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
        return μ, σ2 = conditional(gp, X)
end

# 1D Case for prediction
predict(gp::GPMC, x::Vector{Float64}; full_cov::Bool=false) = predict(gp, x'; full_cov=full_cov)


function get_params(gp::GPMC; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    if lik  && num_params(gp.lik)>0; push!(params, get_params(gp.lik)); end
    if mean;  append!(params, get_params(gp.m)); end
    if kern; append!(params, get_params(gp.k)); end
    return params
end

function set_params!(gp::GPMC, hyp::Vector{Float64}; lik::Bool=true, mean::Bool=true, kern::Bool=true)
    # println("mean=$(mean)")
    gp.v = hyp[1:gp.nobsv]
    if lik  && num_params(gp.lik)>0;; set_params!(gp.lik, hyp[gp.nobsv+1:gp.nobsv+num_params(gp.lik)]); end
    if mean; set_params!(gp.m, hyp[gp.nobsv+1+num_params(gp.lik):gp.nobsv+num_params(gp.lik)+num_params(gp.m)]); end
    if kern; set_params!(gp.k, hyp[end-num_params(gp.k)+1:end]); end
end
    
function push!(gp::GPMC, X::Matrix{Float64}, y::Vector{Float64})
    warn("push! method is currently inefficient as it refits all observations")
    if gp.nobsv == 0
        GaussianProcesses.fit!(gp, X, y)
    elseif size(X,1) != size(gp.X,1)
        error("New input observations must have dimensions consistent with existing observations")
    else
        GaussianProcesses.fit!(gp, cat(2, gp.X, X), cat(1, gp.y, y))
    end
end

push!(gp::GPMC, x::Vector{Float64}, y::Vector{Float64}) = push!(gp, x', y)
push!(gp::GPMC, x::Float64, y::Float64) = push!(gp, [x], [y])
push!(gp::GPMC, x::Vector{Float64}, y::Float64) = push!(gp, reshape(x, length(x), 1), [y])

function show(io::IO, gp::GPMC)
    println(io, "GP object:")
    println(io, "  Dim = $(gp.dim)")
    println(io, "  Number of observations = $(gp.nobsv)")
    println(io, "  Mean function:")
    show(io, gp.m, 2)
    println(io, "  Kernel:")
    show(io, gp.k, 2)
    if (gp.nobsv == 0)
        println("  No observation data")
    else
        println(io, "  Input observations = ")
        show(io, gp.X)
        print(io,"\n  Output observations = ")
        show(io, gp.y)
        if typeof(gp.lik)!=Gaussian
            print(io,"\n  Log-Likelihood = ")
            show(io, round(gp.ll,3))
        else
            print(io,"\n  Marginal Log-Likelihood = ")
            show(io, round(gp.mLL,3))

        end            
    end
end

