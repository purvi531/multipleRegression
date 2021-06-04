# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 18:57:04 2021

@author: HP
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston  #online importing file from sklearn
boston=load_boston()
boston.keys()
boston['data'].shape
bos=pd.DataFrame(boston.data)  #it will take only data values(to store it as a dataframe)
print(boston.feature_names)  #these are features  inside feature_names key
print(boston.DESCR)  #it is a desription of dataset inside DESCR key
print(boston.target)  #it is label

bos.columns=boston.feature_names  #giving name to coloums insteed  of indexing
print(bos)

bos['price']=boston.target   #adding lable coloum in is bos dataframe
print(bos)


x=bos.drop('price',axis=1)  #means complete coloum, axis=1 means column
y=bos['price']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=1)
model=LinearRegression() #y=a1x1+a2x2+.....a13x13+b
#fitting data into model
mynewmodel=model.fit(xtrain,ytrain)
y_pred_test=mynewmodel.predict(xtest)
y_pred_train=mynewmodel.predict(xtrain)

df=pd.DataFrame(ytest,y_pred_test)


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(ytest,y_pred_test) 


#vizualization
plt.scatter(ytest,y_pred_test,c='red',marker='+')
plt.plot(xtest,mynewmodel.predict(xtest))
plt.show()