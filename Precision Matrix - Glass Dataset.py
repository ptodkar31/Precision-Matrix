# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:19:40 2024

@author: Priyanka
"""

"""
Problem statement-
A glass manufacturing plant uses different earth elements to design 
new glass materials based on customer requirements. For that, they would 
like to automate the process of classification as it’s a tedious job to manually 
classify them. Help the company achieve its objective by correctly classifying 
the glass type based on the other features using KNN algorithm.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

glass=pd.read_csv("C:\Data Set\glass.csv")
 
#to split train and test data
from sklearn.model_selection import train_test_split
train,test=train_test_split(glass,test_size=0.3,random_state=0)

#KNN
from sklearn.neighbors import KNeighborsClassifier as KNC
#to find best k value
acc=[]
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])
    
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'bo-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'ro-')
plt.legend(['train','test'])


#from plots at k=5 we get best model
#model building at k=5 
neigh=KNC(n_neighbors=5)
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
pred_train=neigh.predict(train.iloc[:,0:9])
train_acc=np.mean(pred_train==train.iloc[:,9])
train_acc#0.76
pred_test=neigh.predict(test.iloc[:,0:9])
test_acc=np.mean(pred_test==test.iloc[:,9])
test_acc#0.661
