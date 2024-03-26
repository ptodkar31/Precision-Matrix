# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:24:26 2024

@author: Priyanka
"""

"""problem statement-
A National Zoopark in India is dealing with the problem of 
segregation of the animals based on the different attributes 
they have. Build a KNN model to automatically classify the animals.
 Explain any inferences you draw in the documentation
 """
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Zoo=pd.read_csv("C:\Data Set\Zoo.csv")
zoo=Zoo.iloc[:,1:]

#to split train and test data
from sklearn.model_selection import train_test_split
train,test=train_test_split(zoo,test_size=0.3,random_state=0)

#KNN
from sklearn.neighbors import KNeighborsClassifier as KNC

#to find best k value
acc=[]
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
    test_acc=np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
    acc.append([train_acc,test_acc])
    
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'bo-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'ro-')
plt.legend(['train','test'])

#from plots atk=5 we get best model
#model building at k=5
neigh=KNC(n_neighbors=5)
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
train_acc=np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
test_acc=np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
train_acc
test_acc
