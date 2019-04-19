# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 15:49:00 2018

@author: igorl
"""

import requests
import pandas
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#carrega os dados
fileTrain = pandas.read_csv('train.csv')
fileTest = pandas.read_csv('test3.csv')

dataTrain = fileTrain[['NU_INSCRICAO', 'TX_RESPOSTAS_MT', 'CO_PROVA_MT']]
dataTrain.fillna(value='.............................................', inplace=True)
dataTest = fileTest[['NU_INSCRICAO', 'TX_RESPOSTAS_MT', 'CO_PROVA_MT']]
dataTest.fillna(value='........................................', inplace=True)

#trata os dados
#train
s_train = pandas.Series(dataTrain.iloc[:,1])
respostas_train = s_train.str.extract('([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])', expand=True)
cat_train = fileTrain[['TP_ESCOLA','CO_PROVA_CN','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','Q001','Q002','Q006', 'Q024','Q025','Q026','Q027','Q047']]
dummies_train =  pandas.concat([cat_train, respostas_train.iloc[:, :40]], axis=1, ignore_index=True)

#ytrain
y = respostas_train.iloc[:, 40:].replace(['A', 'B', 'C', 'D', 'E', '.', '*'], [1,2,3,4,5, 6,6])
y1_train = y.iloc[:,0]
y2_train = y.iloc[:,1]
y3_train = y.iloc[:,2]
y4_train = y.iloc[:,3]
y5_train = y.iloc[:,4]

#test
s_teste = pandas.Series(dataTest.iloc[:,1])
respostas_test = s_teste.str.extract('([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])', expand=True)
cat_test = fileTest[['TP_ESCOLA','CO_PROVA_CN','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','Q001','Q002','Q006', 'Q024','Q025','Q026','Q027','Q047']]
dummies_test = pandas.concat([cat_test, respostas_test.iloc[:, :40]], axis=1, ignore_index=True)


dummies = pandas.concat([dummies_train, dummies_test], axis=0, ignore_index=True)
dummies = pandas.get_dummies(dummies, drop_first=True)

dummies_train = dummies.iloc[:13452,:]
dummies_test = dummies.iloc[13452:,:]

#more features
features_train = fileTrain[['NU_IDADE', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']]
features_train.fillna(value=0, inplace=True)
features_test = fileTest[['NU_IDADE', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']]
features_test.fillna(value=0, inplace=True)

#final features
x_train = pandas.concat([features_train, dummies_train], axis=1, ignore_index=True)
features_test.reset_index(drop=True, inplace=True)
dummies_test.reset_index(drop=True, inplace=True)
x_test = pandas.concat([features_test, dummies_test], axis=1, ignore_index=True)




#PRE-PROCESSING FEATURES

sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train)
a = pca.explained_variance_ratio_

# Fitting Kernel SVM to the Training set
#classifier1 = SVC(kernel = 'rbf', C = 100)
#classifier2 = SVC(kernel = 'rbf', C = 100)
#classifier3 = SVC(kernel = 'rbf', C = 100)
#classifier4 = SVC(kernel = 'rbf', C = 100)
#classifier5 = SVC(kernel = 'rbf', C = 100)


steps = [ ('SVC', SVC(kernel='rbf'))]
param_grid = {'SVC__C': [10, 50, 100, 200, 1000]}
estimator = Pipeline(steps=steps)
classifier1 = GridSearchCV(estimator, param_grid, verbose=5, n_jobs=-1, cv=10)
classifier2 = GridSearchCV(estimator, param_grid, verbose=5, n_jobs=-1, cv=10)
classifier3 = GridSearchCV(estimator, param_grid, verbose=5, n_jobs=-1, cv=10)
classifier4 = GridSearchCV(estimator, param_grid, verbose=5, n_jobs=-1, cv=10)
classifier5 = GridSearchCV(estimator, param_grid, verbose=5, n_jobs=-1, cv=10)

classifier1.fit(x_train, y1_train)
classifier2.fit(x_train, y2_train)
classifier3.fit(x_train, y3_train)
classifier4.fit(x_train, y4_train)
classifier5.fit(x_train, y5_train)






#PREDICTION
#predict + convertendo as respostas em letras
y1_test = pandas.DataFrame(data = classifier1.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','.'])
y2_test = pandas.DataFrame(data = classifier2.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','.'])
y3_test = pandas.DataFrame(data = classifier3.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','.'])
y4_test = pandas.DataFrame(data = classifier4.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','.'])
y5_test = pandas.DataFrame(data = classifier5.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','.'])

#string final
y_test =  y1_test.astype(str) + y2_test.astype(str) + y3_test.astype(str) + y4_test.astype(str)+ y5_test.astype(str)





#resposta em HTTP
result =  pandas.concat([dataTest['NU_INSCRICAO'], y_test], axis=1, ignore_index=True)
result.columns = ['NU_INSCRICAO', 'TX_RESPOSTAS_MT']

answers = result.to_dict('records')
response = {'token':'efb1548e92ce3b7afba62f4d7f594ffd1e743ff7', 'email':'igor@gmail.com', 'answer':answers}

r = requests.post("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit", data = json.dumps(response))




r.text
