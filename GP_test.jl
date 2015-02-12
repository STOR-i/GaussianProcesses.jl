using Distributions
using Gadfly
# using Winston

x_pred = [-5:0.1:5]        # Define points at which to define function
l = 1.0                    # Scale length

# Define covariance function (in this case we are using the squared exponential)
function exp_cov(Xi::Vector{Float64}, Xj::Vector{Float64}, l::Float64=1.0)
    Sigma = Array(Float64, length(Xi),length(Xj))
    for i=1:size(Sigma,1)
        for j=1:size(Sigma,2)
            Sigma[i,j] = exp(-0.5/l^2*(Xi[i]-Xj[j])^2)
        end
    end
    return Sigma
end

# Specify some observations
x_obs = [-4.0,-3.0,-1.0, 0.0, 2.0] 
y = [-2.0, 0.0, 1.0, 2.0, -1.0]

# Fit GP to data
cov_xx_inv = inv(exp_cov(x_obs,x_obs))
eF = exp_cov(x_pred,x_obs)*cov_xx_inv*y   # Mean function

# Note that numerical errors are causing some diagonal elements to have negligibly small negative values
cF = exp_cov(x_pred,x_pred) - exp_cov(x_pred,x_obs) *cov_xx_inv * exp_cov(x_obs,x_pred)  # Covariance function
u = eF  + 2*sqrt(max(diag(cF),0.0)) # Lower bound
l = eF  - 2*sqrt(max(diag(cF),0.0)) # Upper bound

# Gadfly plot results
plot(x=x_pred, y=eF, ymin=l, ymax=u, Geom.line, Geom.ribbon)

### Winston plot
## p = plot(x_pred,eF)
## oplot(x_obs,y,"*")
## oplot(x_pred,eF-2*sqrt(max(diag(cF),0.0))),"b")
## oplot(x_pred,eF+2*sqrt(max(diag(cF),0.0))),"b")
