# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:57:27 2018

@author: igorl
"""

import requests
import pandas
import numpy as np
import json
import matplotlib.pyplot as plt

fileTrain = pandas.read_csv('train.csv')
fileTest = pandas.read_csv('test.csv')

dataTrain = fileTrain[['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO', 'NU_NOTA_COMP4', 'NU_NOTA_MT']]

dataTrain.fillna(value=0, inplace=True)
x_train = dataTrain.iloc[:,1:-1].values
y_train = dataTrain.iloc[:, -1].values
          
categorical_data = fileTrain[['Q006','Q025','Q047']]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
labelencoder2 = LabelEncoder()
labelencoder3 = LabelEncoder()
categorical_data.iloc[:, 0] = labelencoder.fit_transform(categorical_data.iloc[:, 0])
categorical_data.iloc[:, 1] = labelencoder2.fit_transform(categorical_data.iloc[:, 1])
categorical_data.iloc[:, 2] = labelencoder3.fit_transform(categorical_data.iloc[:, 2])
categorical_data = categorical_data.values
x_train = np.append(arr =x_train, values = categorical_data, axis =1)
      
#SCALING + REGRESSOR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(x_train)

# Fitting SVR to the dataset
# =============================================================================
# from sklearn.svm import SVR
# regressor = SVR(kernel = 'rbf', C=100)
# =============================================================================
# =============================================================================
# 
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()


regressor.fit(X, y_train)








#-------------------------------------------------------------
#TESTE

dataTest = fileTest[['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO', 'NU_NOTA_COMP4']]
dataTest.fillna(value=0, inplace=True)
x_test = dataTest.iloc[:,1:].values


categorical_data_test = fileTest[['Q006','Q025','Q047']]
categorical_data_test.iloc[:, 0] = labelencoder.transform(categorical_data_test.iloc[:, 0])
categorical_data_test.iloc[:, 1] = labelencoder2.transform(categorical_data_test.iloc[:, 1])
categorical_data_test.iloc[:, 2] = labelencoder3.transform(categorical_data_test.iloc[:, 2])
categorical_data_test = categorical_data_test.values
x_test = np.append(arr =x_test, values = categorical_data_test, axis =1)

#PREDICTION
pred = regressor.predict(sc_X.transform(x_test))
pred_DF = pandas.DataFrame(data=pred)
pred_DF[pred_DF<0] = 0

dataTest.reset_index(drop=True, inplace=True)

result =  pandas.concat([dataTest['NU_INSCRICAO'], pred_DF], axis=1, ignore_index=True)
result.columns = ['NU_INSCRICAO', 'NU_NOTA_MT']





answers = result.to_dict('records')
response = {'token':'efb1548e92ce3b7afba62f4d7f594ffd1e743ff7', 'email':'igor@gmail.com', 'answer':answers}



r = requests.post("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-2/submit", data = json.dumps(response))


