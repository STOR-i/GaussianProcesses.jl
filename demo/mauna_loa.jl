#This example fits a GP regression model to the Mauna Loa CO2 data set. This data set is regularly updated and can be found at co2now.org/images/stories/data/co2-mlo-monthly-noaa-esrl.xls
#
#This example follows from Chapter 5 of Gaussian Processes for Machine Learning, Rasmussen and Williams (2006)
############################################################################################

using Gadfly, GaussianProcesses

data = readcsv("CO2_data.csv")

year = data[:,1]; co2 = data[:,2];
x = year[year.<2004]; y = co2[year.<2004];
xpred = year[year.>=2004]; ypred = co2[year.>=2004];

mConst = MeanConst(mean(y))       #Fit the constant mean function

#Kernel is represented as a sum of kernels
kernel = SE(4.0,4.0) + Periodic(0.0,1.0,0.0)*SE(4.0,0.0) + RQ(0.0,0.0,-1.0) + SE(-2.0,-2.0)

gp = GP(x,y,mConst,kernel,-2.0)   #Fit the GP

plot(gp,clim=(2004.0,2024.0))  #Gadfly can take a while to load


