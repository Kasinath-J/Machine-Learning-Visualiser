import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

size = 100

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
    X[:,i] = temp


#cost function
def cost(t1,t2):
    ans= np.empty(shape=size)
    for i in range(size):
        cost = 0
        for j in range(m):
            y_pred = np.dot(np.array([t1[i],t2[i]]),X[:,j])
            cost =  cost + (y_pred - y[j])**2
        cost = cost/(2*m)
        ans[i] = cost
    return ans

theta0 = np.linspace(-1000,1000,size)
theta1 = np.linspace(-1000,1000,size)
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





