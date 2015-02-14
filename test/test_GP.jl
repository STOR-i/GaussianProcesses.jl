using gaussianprocesses

# 1-dimensional case
xpred = [-4.0,-3.0,-1.0, 0.0, 2.0]
y = [-2.0, 0.0, 1.0, 2.0, -1.0]
gp = GP(xpred,y,rbf)
predict(gp, xpred)

# 2-dimensional case
xpred = [[0.0, 0.0] [1.0, 0.0] [0.0, 1.0] [1.0, 1.0] [0.5, 0.5]]
y = [-2.0, 0.0, 1.0, 2.0, -1.0]
gp = GP(xpred,y,rbf)
predict(gp, xpred)
