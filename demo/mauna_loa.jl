#This example fits a GP regression model to the Mauna Loa CO2 data set. This data is available from co2now.org/images/stories/data/co2-mlo-monthly-noaa-esrl.xls
#
#This example follows from Chapter 5 of Gaussian Processes for Machine Learning, Rasmussen and Williams (2006)
############################################################################################

using Gadfly, GaP

data = readcsv("CO2_data.csv")

year = data[:,1]; co2 = data[:,2]
x = year[year.<2004]; y = co2[year.<2004]
xpred = year[year.>=2004]; ypred = co2[year.>=2004]


kernel = SE(4.0,4.0) + Peri(0.0,1.0,0.0)*SE(4.0,0.0) + RQ(0.0,0.0,-1.0) + SE(-2.0,-2.0)

mConst = MeanConst(mean(y))
gp = GP(x,y,mConst,kernel,-2.0)
optimize!(gp,method=:bfgs,show_trace=true)

mu, Sigma = predict(gp,xpred)

plot(gp,xpred)


