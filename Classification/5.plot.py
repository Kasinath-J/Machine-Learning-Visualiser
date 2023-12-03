import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

points = [[1,1,0],[2,2,0],[3.5,3.5,0],[4,4,1],[5,5,1],[6,6,1]]
m = len(points)

size = 20

#learning rate
rate = 0.3

# X values appended with 1 and y values
X = np.empty(shape=[3,m])
y = np.empty(shape=[1,m])


for i in range(m):
    temp = np.insert(np.delete(points[i],-1,0),0,1)
    y[0][i] = points[i][2]
    temp = temp.transpose()
    X[:,i] = temp

#initial parameters
theta = np.zeros(shape=(1,3))

#cost function
def cost():
    y_pred = 1/(1+np.exp(-np.dot(theta,X)))
    cost = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/m
    print("cost",cost)
    return cost

X1 = np.linspace(min(X[1]-1), max(X[1])+1,size)
X2 = np.linspace(min(X[2]-1), max(X[2])+1,size)
X1,X2 = np.meshgrid(X1,X2)

def pred(x1,x2):
    ans = np.empty(shape=size)
    for i in range(size):
        y_pred = 1/(1+np.exp(-np.dot(theta,np.array([1,x1[i],x2[i]]))))
        ans[i] = y_pred
    return ans

#updating theta until cost = 0   
y_all = np.empty(shape=(2000,size,size))
i=0
while(cost()>=0.1 and i<2000):    
    y_pred = 1/(1+np.exp(-np.dot(theta,X)))
    diff = np.dot((y_pred-y),X.transpose())
    
    y_test = np.array(list(map(pred,X1,X2)))
    theta = theta - rate/m*diff
    y_all[i] = y_test
    i+=1

y_all.reshape(i,size,size)
##for plotting

fig = plt.figure( figsize=(25,25))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[1],X[2],y)
plot = [ax.plot_surface(X1, X2, y_all[0], color='0.75')]
ax.set_zlim(-0.5,1.5)

def update_plot(frame_number, y_all, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X1, X2, y_all[frame_number], cmap="magma")
    print(frame_number)

animate = animation.FuncAnimation(fig, update_plot, len(y_all), interval = 1,fargs=(y_all, plot))
plt.show()

