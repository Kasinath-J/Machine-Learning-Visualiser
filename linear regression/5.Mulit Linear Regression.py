import numpy as np
import matplotlib.pyplot as plt

# Actual X and y values
points = np.array([[1,1,2],[2,2,4],[3,3,6],[4,4,8],[5,5,10]])
m = len(points)

#learning rate
rate = 0.001

# X values appended with 1 and y values
X = np.empty(shape=[3,m])
y = np.empty(shape=[m])


for i in range(m):
    temp = np.insert(np.delete(points[i],-1,0),0,1)
    y[i] = points[i][2]
    temp = temp.transpose()
    X[:,i] = temp

#initial parameters
theta = np.zeros(shape=[3])

#cost function
def cost():
    cost = 0
    for i in range(m):
        y_pred = np.dot(theta,X[:,i])
        cost =  cost + (y_pred - y[i])**2

    cost = cost/(2*m)
    print("cost",cost)
    return cost


#updating theta until cost = 0
#for practice lets keep theta0 = 0

while(cost()>=0.001):    
    diff0 = 0
    diff1 = 0
    diff2 = 0
    for i in range(m):
        y_pred = np.dot(theta,X[:,i])
        diff0 = diff0 + (y_pred - y[i])
        diff1 =  diff1 + (y_pred - y[i])*X[1][i]
        diff2 = diff2 +  (y_pred - y[i])*X[2][i]
        
    theta[0] = theta[0] - rate/m*diff0
    theta[1] = theta[1] - rate/m*diff1
    theta[2] = theta[2] - rate/m*diff2
    
    print("thetas : ",theta[0]," ",theta[1]," ",theta[2])

print("final theta1",theta[0],theta[1],theta[2])




