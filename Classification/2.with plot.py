import numpy as np
import matplotlib.pyplot as plt

points = [[1,0],[2,0],[3.5,0],[4,1],[5,1],[6,1]]
m = len(points)

#learning rate
rate = 0.5

# X values appended with 1 and y values
X = np.empty(shape=[2,m])
y = np.empty(shape=[1,m])


for i in range(m):
    temp = np.insert(np.delete(points[i],1,0),0,1)
    y[0][i] = points[i][1]
    temp = temp.transpose()
    #X[0][i] = temp[0]
    #X[1][i] = temp[1]
    X[:,i] = temp

#initial parameters
theta = np.zeros(shape=(1,2))

#cost function
def cost():
    y_pred = 1/(1+np.exp(-np.dot(theta,X)))
    cost = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/m
    print("cost",cost)
    return cost


#updating theta until cost = 0

#for plotting
plt.ion()
fig,ax = plt.subplots(figsize=(8,5))
ax.set_xlim(min(X[1])-1, max(X[1])+1)
ax.set_ylim(-2,2)
line1,= ax.plot(X[1], np.zeros(m),color = 'r')
scatter1= ax.scatter(X[1],y,color = 'g')
plt.title("Logistic Regression", fontsize=15)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")


   
while(cost()>=0.2):    
    y_pred = 1/(1+np.exp(-np.dot(theta,X)))
    diff = np.dot((y_pred-y),X.transpose())
    
    X_test = np.linspace(-min(X[1])-1,max(X[1])+1,100).reshape(1,-1)
    line1.set_xdata(X_test)
    X_test = np.vstack((np.ones(shape = (1,100)),X_test))
    y_test = 1/(1+np.exp(-np.dot(theta,X_test)))
    line1.set_ydata(y_test)
    
    theta = theta - rate/m*diff
    
    print(np.append(X[1],0),np.append(y_pred,np.dot(theta,[1,0])))
    fig.canvas.draw()
    fig.canvas.flush_events()
    #time.sleep(0.2)

print("final theta1",theta)







