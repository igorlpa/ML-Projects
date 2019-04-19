# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:18:50 2018

@author: igor
"""

import requests
from joblib import Parallel, delayed
import pandas
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


#READ DATA
features      = ['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO', 'NU_NOTA_COMP1','NU_NOTA_COMP2','NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_MT']
features_test = ['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO', 'NU_NOTA_COMP1','NU_NOTA_COMP2','NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']
features_cat = ['TP_ESCOLA','CO_PROVA_CN','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','Q001','Q002','Q006','Q024','Q025','Q026','Q027','Q047']


#TRAIN
fileTrain = pandas.read_csv('train.csv')

dataTrain = fileTrain[features]
dataTrain.fillna(value=0, inplace=True)
x_train = dataTrain.iloc[:,1:-1].values
y_train = dataTrain.iloc[:, -1].values


categorical_data = fileTrain[features_cat]
categorical_data.fillna(value=0, inplace=True)

#TESTE
fileTest = pandas.read_csv('test2.csv')
dataTest = fileTest[features_test]
dataTest.fillna(value=0, inplace=True)
x_test = dataTest.iloc[:,1:].values

categorical_data_test = fileTest[features_cat]
categorical_data_test.fillna(value=0, inplace=True)


#dummies
cat =  pandas.concat([categorical_data, categorical_data_test], axis=0, ignore_index=True)
dummies = pandas.get_dummies(cat, drop_first=True)

dummies_train = dummies.iloc[:13452,:]
dummies_test = dummies.iloc[13452:,:]

x_train = np.append(arr = x_train, values = dummies_train.values, axis =1)
x_test = np.append(arr = x_test, values = dummies_test.values, axis =1)


#SCALING + REGRESSOR
sc_X = StandardScaler()
X = sc_X.fit_transform(x_train)

# Fitting SVR to the dataset


# =============================================================================
# regressor = SVR(kernel = 'rbf', C=300)
# regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
# regressor  = LinearRegression()
# =============================================================================

# =============================================================================
#     steps = [('SKB', SelectKBest(score_func= f_regression )),  ('SVR', SVR(kernel='rbf'))]
#     param_grid = {'SVR__C': [100, 500, 600, 800, 1000], 'SKB__k':[2,3,4,5,6,7,8]}
# =============================================================================

# =============================================================================
# steps = [('SKB', SelectKBest(score_func= f_regression )),  ('RF', RandomForestRegressor())]
# param_grid = {'RF__n_estimators': [5, 10, 20, 30], 'SKB__k':[2,3,4,5,6,7,8,9]}
# =============================================================================

steps = [('SKB', SelectKBest(score_func= f_regression )),  ('LR', LinearRegression())]
param_grid = {'LR__fit_intercept': [True, False], 'SKB__k':[2,3,4,5,6,7,8,9, 10, 15, 20, 30, 40, 50, 60, 80, 90]}


print("teste")
estimator = Pipeline(steps=steps)
cv = GridSearchCV(estimator, param_grid, verbose=5, n_jobs=-1, cv=10)
cv.fit(X, y_train)

print(cv.best_params_, cv.best_score_)



#PREDICTION

pred = cv.predict(sc_X.transform(x_test))
pred_DF = pandas.DataFrame(data=pred)
pred_DF[pred_DF<0] = 0

dataTest.reset_index(drop=True, inplace=True)

result =  pandas.concat([dataTest['NU_INSCRICAO'], pred_DF], axis=1, ignore_index=True)
result.columns = ['NU_INSCRICAO', 'NU_NOTA_MT']
answers = result.to_dict('records')
response = {'token':'efb1548e92ce3b7afba62f4d7f594ffd1e743ff7', 'email':'igor@gmail.com', 'answer':answers}



# r = requests.post("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-2/submit", data = json.dumps(response))


    