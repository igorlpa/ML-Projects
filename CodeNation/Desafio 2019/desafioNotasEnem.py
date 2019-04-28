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

features = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']
features_cat = ['CO_PROVA_CN','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT']

#TRAIN
dataTrain = fileTrain[features]
dataTrain.fillna(value=0, inplace=True)
x_train = dataTrain.iloc[:,:].values
y_train = fileTrain[['NU_NOTA_MT']].fillna(value=0)

categorical_train = fileTrain[features_cat]
categorical_train.fillna(value=0, inplace=True)

          

#TESTE
dataTest = fileTest[features]
dataTest.fillna(value=0, inplace=True)
x_test = dataTest.iloc[:,:].values

categorical_test = fileTest[features_cat]
categorical_test.fillna(value=0, inplace=True)

#dummies
cat =  pandas.concat([categorical_train, categorical_test], axis=0, ignore_index=True)
dummies = pandas.get_dummies(cat, drop_first=True)

dummies_train = dummies.iloc[:categorical_train.shape[0],:]
dummies_test = dummies.iloc[categorical_train.shape[0]:,:] 


#jutando as features
x_train = np.append(arr = x_train, values = dummies_train.values, axis =1)
x_test = np.append(arr = x_test, values = dummies_test.values, axis =1)





#SCALING + REGRESSOR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(x_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', C=100)
# =============================================================================
# 
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
# =============================================================================

regressor.fit(X, y_train)








#-------------------------------------------------------------


#PREDICTION
pred = regressor.predict(sc_X.transform(x_test))
pred_DF = pandas.DataFrame(data=pred)

dataTest.reset_index(drop=True, inplace=True)

result =  pandas.concat([fileTest['NU_INSCRICAO'], pred_DF], axis=1, ignore_index=True)
result.columns = ['NU_INSCRICAO', 'NU_NOTA_MT']

result.to_csv('answer.csv', index=False)
