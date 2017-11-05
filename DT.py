# Required Python Packages
import numpy as np
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt

#Arbitrary Data creation
np.random.seed(19680801)

samples = 1000
inp1 = 6*np.random.rand(samples,1) - 3
inp2 = 6*np.random.rand(samples,1) - 3
y = np.zeros(samples)
for i in range(0,inp1.size,1):
    if (inp2[i] > 2)or((inp2[i] > -2)and(inp1[i] > 1))or((inp2[i] > 0)and(inp1[i] >= -1)and(inp1[i] <= 1)):
        y[i] = 1
    else:
        y[i] = 0
Xtrain = np.concatenate((inp1,inp2),axis=1)


model = tree.DecisionTreeClassifier()
model.fit(Xtrain, y)

#plot data
for i in range(0,inp1.size,1):
    if y[i] == 1:
        plt.scatter(inp1[i],inp2[i],color = "r")
    else:
        plt.scatter(inp1[i],inp2[i],color = "g")

#plot decision boundary
plot_step = 0.02
inp1_min, inp1_max = inp1.min(), inp1.max()
inp2_min, inp2_max = inp1.min(), inp1.max()
xx,yy = np.meshgrid(np.arange(inp1_min,inp1_max,plot_step),
                    np.arange(inp2_min,inp2_max,plot_step))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
c = ("#ffddcc","#ccffee","#003300")
cs = plt.contour(xx, yy, Z, s=0.5)
plt.show()

#test decision tree
Xtest = [[1, 3]]
print(model.predict(Xtest))

#save decision tree
with open("quant_classifier.txt", "w") as f:
    f = tree.export_graphviz(model, out_file=f)
