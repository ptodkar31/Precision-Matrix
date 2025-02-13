# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:09:48 2024

@author: Priyanka
"""


import pandas as pd
import numpy as np
wbcd=pd.read_csv("C:\Data Set\wbcd.csv")
#there are 569 rows and 32 columns
wbcd.describe()
#in output column there is only B for Benies ans M for malignant
#let us first convert it as benies and maliganat
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='B','Beniegn',wbcd['diagnosis'])
#in wbcd there is column named diagnosis where ever there is B replace with Benign
#similarly where ever there is M in the same column replace with "malignant"
wbcd['diagnosis']=np.where(wbcd['diagnosis']=='M','Malignanat',wbcd['diagnosis'])


#0th column is patient ID let us Drop It
wbcd=wbcd.iloc[:,1:32]

#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now let us apply this function to the dataframe
wbcd_n=norm_func(wbcd.iloc[:,1:32])
#because now 0th column is output or label it is not considerd hence 1:
    
#Let us now apply x as input and y as output
x=np.array(wbcd_n.iloc[:,:])
#since in wbcd_n we are already excludeing output column hence all rows and columns
y=np.array(wbcd['diagnosis'])


#now let us split the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#here you are passing x,y instead dataframe handle
#there colud chances of unbalancing of data
#let us assume you have 100 data points outof which 80 Not cancer and 20 cancer
#these data points must be equally distributed
#there is statified sampling concept is used

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
pred
#now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))
pd.crosstab(pred,y_test)

#let us check the apllicability of the model
#that means classification actual patient is maligenant
#cancer patient but predicted is Benien is 1
#actual patient is Benien and Predicted as cancer patient is 5
#hence this model is not acceptable


#let us try to select correct values of k
acc=[]
#Running KNN algorithm for k=3 to 50 inthe step of 2
#k value selected is odd value
for i in range(3,50,2):
    #Declare the model
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train,y_train)
    train_acc=np.mean(neigh.predict(x_train)==y_train)
    test_acc=np.mean(neigh.predict(x_test)==y_test)
    acc.append([train_acc,test_acc])
#if you will see the acc it has got two accuracy,i[0]-train_acc
#i[1]=test_acc
#to plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
#there are 3,5,7,9 are possible values where accuracy is goog 
#let us check for k=3
knn=KNeighborsClassifier(n_neighbors=7)
knn.fitt(x_train,y_train)
pred=knn.predict(x_test)
accuracy_score(pred,y_test)
pd.crosstab(pred,y_test)

#miss classification actual patient is malignant
#i.e cancer patient but predicted is benien is 1
#Actual opatient is Beninen and predicted as cancer patient is 2
#hence this model is not acceptable for 5 same sinario
#for k=7 we are getting zero false positive and good accuracy
#hence k=7 is appropriate values of k

