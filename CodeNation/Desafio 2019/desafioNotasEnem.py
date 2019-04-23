# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:00:43 2019

@author: igorl
"""

import pandas
import numpy as np
import json
import matplotlib.pyplot as plt

fileTrain = pandas.read_csv('train.csv')
fileTest = pandas.read_csv('test.csv')

dataTrain = fileTrain[['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT']]

dataTrain.fillna(value=0, inplace=True)

x_train = dataTrain.iloc[:,1:-1].values

y_train = dataTrain.iloc[:, -1].values
          




      
#SCALING + REGRESSOR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(x_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', C=400)
# =============================================================================
# 
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
# =============================================================================

regressor.fit(X, y_train)








#-------------------------------------------------------------
#TESTE

dataTest = fileTest[['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC']]
dataTest.fillna(value=0, inplace=True)

x_test = dataTest.iloc[:,1:].values


#PREDICTION
pred = regressor.predict(sc_X.transform(x_test))
pred_DF = pandas.DataFrame(data=pred)

dataTest.reset_index(drop=True, inplace=True)

result =  pandas.concat([dataTest['NU_INSCRICAO'], pred_DF], axis=1, ignore_index=True)
result.columns = ['NU_INSCRICAO', 'NU_NOTA_MT']

result.to_csv('answer.csv', index=False)
