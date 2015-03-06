# Contour plots

using gaussianprocesses
using Gadfly
import Gadfly.plot


#Training data

## x = 2*π*rand(5);
## y = cos(x) + 0.5*randn(5);

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];

#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1
se = SE()

gp = GP(x,y,meanZero,se)
optimize!(gp)


sigmas = linspace(0.1,2,100)
ells   = linspace(0.1,2,100)
cont   = zeros(100,100)
 
for i in 1:100, j in 1:100
    se = SE(log(ells[j]),log(sigmas[i]))
    gp = GP(x,y,meanZero,se)
    cont[i,j] = gp.mLL
end

plot(z=cont,x=log10(ells),y=log10(sigmas),Geom.contour)
    



