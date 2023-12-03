import numpy as np
import matplotlib.pyplot as plt

points = [[1,0],[2,0],[3.5,0],[4,1],[5,1],[6,1]]
m = len(points)

#learning rate
rate = 0.1

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
   
while(cost()>=0.2):    
    y_pred = 1/(1+np.exp(-np.dot(theta,X)))
    diff = np.dot((y_pred-y),X.transpose())
    theta = theta - rate/m*diff
    #print("theta",theta)   
    print(np.append(X[1],0),np.append(y_pred,np.dot(theta,[1,0])))
    
print("final theta1",theta)


