import numpy as np
import time
import matplotlib.pyplot as plt

# Actual X and y values
points = np.array([[1,1],[2,4],[3,6],[4,8],[5,10]])
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
theta = np.zeros(2)

#cost function
def cost():
    cost = 0
    for i in range(m):
        y_pred = np.dot(theta,X[:,i])
        cost =  cost + (y_pred - y[i])**2

    cost = cost/(2*m)
    return cost

c = []
for i in range(-10,10,1):
    theta[1] = i;
    c.append(cost())

plt.plot(np.arange(-len(c)/2,len(c)/2),c)
plt.show()

    
    



    

