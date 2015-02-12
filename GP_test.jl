using Distributions
using Winston

x_pred = [-5:0.1:5]        # Define points at which to define function
l = 1.0                    #Scale length

#Define covariance function (in this case we are using the squared exponential)
function COV(Xi::Array{Float64,1},Xj::Array{Float64,1},l::Float64=1.0)
    Sigma = zeros(length(Xi),length(Xj))
    for i=1:size(Sigma)[1]
        for j=1:size(Sigma)[2]
            Sigma[i,j] = exp(-0.5/l^2*(Xi[i]-Xj[j])^2)
        end
    end
    return Sigma
end


#Specify some observations
x_obs = [-4.0,-3.0,-1.0,0.0,2.0] 
y = [-2.0,0.0,1.0,2.0,-1.0]

#Fit GP to data
cov_xx_inv = inv(COV(x_obs,x_obs))
eF = COV(x_pred,x_obs)*cov_xx_inv*y                                       #Mean function
cF = COV(x_pred,x_pred) - COV(x_pred,x_obs)*cov_xx_inv*COV(x_obs,x_pred)  #Covariance function

#Plot results
plot(x_pred,eF)
oplot(x_obs,y,"*")
oplot(x_pred,eF-2*real(sqrt(complex(diag(cF)))),"b")
oplot(x_pred,eF+2*real(sqrt(complex(diag(cF)))),"b")
