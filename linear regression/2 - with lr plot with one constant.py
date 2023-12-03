import numpy as np
import time
import matplotlib.pyplot as plt

# Actual X and y values
points = np.array([[1,100],[2,200],[3,300],[4,400],[5,500]])
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
    #print("cost",cost)
    return cost

#for plotting
plt.ion()
fig,ax = plt.subplots()
ax.set_xlim(0, max(X[1]))
ax.set_ylim(0, max(y))
line1,= ax.plot(X[1], np.zeros(m),color = 'r')
scatter1= ax.scatter(X[1],y,color = 'g')
plt.title("Linear Regression", fontsize=15)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

#updating theta until cost = 0
#for practice lets keep theta0 = 0
while(cost()>=0.01):    
    diff1 = 0
    y_pred = np.empty(shape=m)
    for i in range(m):
        y_pred[i] = np.dot(theta,X[:,i])
        diff1 =  diff1 + (y_pred[i] - y[i])*X[1][i]

    theta[1] = theta[1] - rate/m*diff1
    print(X[1],y_pred)
    
    line1.set_xdata(np.append(X[1],0))
    line1.set_ydata(np.append(y_pred,np.dot(theta,[1,0])))
    fig.canvas.draw()
    fig.canvas.flush_events()
    #time.sleep(0.2)

print("final theta1",theta[1])






