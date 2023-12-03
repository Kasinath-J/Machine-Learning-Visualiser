import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

size = 100

points = [[1,0],[2,0],[3.5,0],[4,1],[5,1],[6,1]]
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
    X[:,i] = temp

#maximum punishment
max_val = max(X[1])*10

#cost function
def cost(t1,t2):
    ans= np.empty(shape=size)
    for i in range(size):    
        y_pred = 1/(1+np.exp(-np.dot(np.array([t1[i],t2[i]]),X)))
        cost = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/m
        #print("cost",cost,"pred",y_pred)
        ans[i] = cost
    ans[np.isnan(ans)] = max_val
    print(ans)
    return ans

theta0 = np.linspace(-10,10,size)
theta1 = np.linspace(-10,10,size)
theta0, theta1 = np.meshgrid(theta0,theta1)
c = list(map(cost,theta0,theta1))
c = np.reshape(c,(size,size))
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0,theta1, c, cmap="plasma")
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('cost')
plt.show()





