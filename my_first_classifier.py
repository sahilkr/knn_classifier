__author__ = 'SAHIL'
import numpy as np
from scipy.spatial import distance
def dis(a,b):
    return distance.euclidean(a,b)



class classifier():
    def fit(self,x_test,y_test):
        self.x_test=x_test
        self.y_test=y_test
    def predict(self,x_target):
        prediction=[]

        for i in range(len(x_target)):
            prediction.append(self.b_res(x_target[i]))
        return prediction
    def b_res(self,x_target):
        dist=dis(x_target,self.x_test[0 ])
        d_index=0
        for j in range(len(self.x_test)):
            distance=dis(x_target,self.x_test[j])
            if dist>distance:
                dist=distance
                d_index=j
        return self.y_test[d_index]
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_test,x_target,y_test,y_target=train_test_split(x,y,test_size=0.5)
#from sklearn import tree
alt=classifier()
alt.fit(x_test,y_test)
predict=alt.predict(x_target)
print(predict)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_target,predict))

