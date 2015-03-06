# Optimize a gaussianprocess

using gaussianprocesses

#Training data

x=[-4.0,-3.0,-1.0,0.0,2.0];
y=[-2.0,0.0,1.0,2.0,-1.0];


#Specify covariance function, not that default hyperparameters are l=1 and sigma²=1

mat52 = MAT52()
gp = GP(x,y,meanZero,mat52)
optimize!(gp, method=:l_bfgs, show_trace=true)

