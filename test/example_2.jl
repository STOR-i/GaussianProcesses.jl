using gaussianprocesses

xpred = [[0.0, 0.0] [1.0, 0.0] [0.0, 1.0] [1.0, 1.0] [0.5, 0.5]]
y = [-2.0, 0.0, 1.0, 2.0, -1.0]

gp = GP(xpred,y,rbf)
predict(gp, xpred)
