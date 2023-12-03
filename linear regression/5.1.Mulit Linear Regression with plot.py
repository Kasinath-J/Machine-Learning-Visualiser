import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# Actual X and y values
points = np.array([[1,1,9],[2,2,8],[3,3,7],[4,4,6],[5,5,5]])
m = len(points)

#learning rate
rate = 0.05

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

#for plotting
plt.ion()
fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, max(X[1]))
ax.set_ylim(0, max(y))
#line1,= ax.plot(X[1],X[2], np.zeros(m),color = 'r')
line1,= ax.plot(0,0,0,color = 'r')
scatter1= ax.scatter(X[1],X[2],y,color = 'g')
plt.title("Linear Regression", fontsize=15)
plt.xlabel("X1")
plt.ylabel("X2")


#updating theta until cost = 0
#for practice lets keep theta0 = 0

while(cost()>=0.01):    
    diff0 = 0
    diff1 = 0
    diff2 = 0
    y_pred = np.empty(shape=m)
    for i in range(m):
        y_pred[i] = np.dot(theta,X[:,i])
        diff0 = diff0 + (y_pred[i] - y[i])
        diff1 =  diff1 + (y_pred[i] - y[i])*X[1][i]
        diff2 = diff2 +  (y_pred[i] - y[i])*X[2][i]

    #line1.set_xdata(np.append(X[1],0))
    #line1.set_ydata(np.append(X[2],0))
    #line1.set_zdata(np.append(y_pred,np.dot(theta,[1,0,0])))
    line1.set_data_3d(np.append(X[1],0),np.append(X[2],0),np.append(y_pred,np.dot(theta,[1,0,0])))
    #line1.set_data_3d(X[1],X[2],y_pred) 
    
    theta[0] = theta[0] - rate/m*diff0
    theta[1] = theta[1] - rate/m*diff1
    theta[2] = theta[2] - rate/m*diff2


    fig.canvas.draw()
    fig.canvas.flush_events()
    #time.sleep(0.1)
    #print("thetas : ",theta[0]," ",theta[1]," ",theta[2])

print("final theta1",theta[0],theta[1],theta[2])




