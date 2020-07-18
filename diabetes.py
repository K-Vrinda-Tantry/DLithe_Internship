# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:52:01 2020

@author: Vrinda
"""
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt 
data=pd.read_csv("diabetes.csv")
data.info()
data.describe()
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

plt.figure(figsize=(20,20))  
sb.heatmap(data.corr(), annot=True, fmt='.0%')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn import preprocessing
pp=preprocessing.StandardScaler()
x=pd.DataFrame(pp.fit_transform(x))

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=812)



from sklearn.linear_model import LogisticRegression as lr
logreg=lr()
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)

#compute the accuracy of prediction
accuracy=logreg.score(x_test,y_test)
print('Logistic Regression accuracy:',accuracy)
