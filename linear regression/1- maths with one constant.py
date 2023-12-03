import numpy as np
import matplotlib.pyplot as plt

# Actual X and y values
points = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
m = len(points)

#learning rate
rate = 0.1

# X values appended with 1 and y values
X = np.empty(shape=[2,m])
y = np.empty(shape=[m])


for i in range(m):
    temp = np.insert(np.delete(points[i],1,0),0,1)
    y[i] = points[i][1]
    temp = temp.transpose()
    #X[0][i] = temp[0]
    #X[1][i] = temp[1]
    X[:,i] = temp

#initial parameters
theta = np.zeros(shape=[2])

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
while(cost()!=0):    
    diff1 = 0
    for i in range(m):
        y_pred = np.dot(theta,X[:,i])
        diff1 =  diff1 + (y_pred - y[i])*X[1][i]
    theta1 = theta[1] - rate/m*diff1
    theta[1] = theta1
    print("theta1",theta1)

print("final theta1",theta[1])




