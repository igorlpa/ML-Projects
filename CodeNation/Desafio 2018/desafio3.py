# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:27:28 2018

@author: igorl
"""

import requests
import pandas
import numpy as np
import json
import matplotlib.pyplot as plt

fileTrain = pandas.read_csv('train.csv')
fileTest = pandas.read_csv('test3.csv')

dataTrain = fileTrain[['NU_INSCRICAO', 'TX_RESPOSTAS_MT', 'CO_PROVA_MT']]
dataTrain.fillna(value='.............................................', inplace=True)

s = pandas.Series(dataTrain.iloc[:,1])
q = s.str.extract('([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])', expand=True)

#tranformando em numeros as respostas
qReplaced = q.replace(['A', 'B', 'C', 'D', 'E', '.', '*'], [1,2,3,4,5, 6,6])
#X = qReplaced.iloc[:, :40]
#adicionando o tio de prova
prova= dataTrain['CO_PROVA_MT']

#transformando o tipo em numero
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Prova = LabelEncoder()
prova = labelencoder_Prova.fit_transform(prova)

qReplaced.reset_index(drop=True, inplace=True)

X = pandas.concat([pandas.DataFrame(data=prova), qReplaced.iloc[:, :40]], axis=1, ignore_index=True)
y1_train = qReplaced.iloc[:,40]
y2_train = qReplaced.iloc[:,41]
y3_train = qReplaced.iloc[:,42]
y4_train = qReplaced.iloc[:,43]
y5_train = qReplaced.iloc[:,44]


onehotencoder = OneHotEncoder()
x_train = onehotencoder.fit_transform(X).toarray()
#normalizando os dados
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#x_train = sc_X.fit_transform(X)
#x_train = X

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', C = 10)
classifier2 = SVC(kernel = 'rbf', C = 10)
classifier3 = SVC(kernel = 'rbf', C = 10)
classifier4 = SVC(kernel = 'rbf', C = 10)
classifier5 = SVC(kernel = 'rbf', C = 10)
# =============================================================================
# from sklearn.ensemble import RandomForestClassifier
# classifier1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier4 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier5 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# =============================================================================


classifier1.fit(x_train, y1_train)
classifier2.fit(x_train, y2_train)
classifier3.fit(x_train, y3_train)
classifier4.fit(x_train, y4_train)
classifier5.fit(x_train, y5_train)


#TESTE

dataTest = fileTest[['NU_INSCRICAO', 'TX_RESPOSTAS_MT', 'CO_PROVA_MT']]
dataTest.fillna(value='........................................', inplace=True)

sTeste = pandas.Series(dataTest.iloc[:,1])
qTeste = sTeste.str.extract('([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])', expand=True)
#tranformando em numeros as respostas
qReplaced = qTeste.replace(['A', 'B', 'C', 'D', 'E', '.', '*'], [1,2,3,4,5, 6,6])
#x_test = qReplaced.iloc[:, :40]
x_test = pandas.concat([dataTest['CO_PROVA_MT'], qReplaced.iloc[:, :40]], axis=1, ignore_index=True)
x_test.iloc[:, 0] = labelencoder_Prova.transform(x_test.iloc[:, 0])

#categorical variables
x_test = onehotencoder.transform(x_test).toarray()

#predict + convertendo as respostas em letras
y1_test = pandas.DataFrame(data = classifier1.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','*'])
y2_test = pandas.DataFrame(data = classifier2.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','*'])
y3_test = pandas.DataFrame(data = classifier3.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','*'])
y4_test = pandas.DataFrame(data = classifier4.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','*'])
y5_test = pandas.DataFrame(data = classifier5.predict(x_test)).replace([1,2,3,4,5, 6], ['A', 'B', 'C', 'D', 'E','*'])

#string final
y_test =  y1_test.astype(str) + y2_test.astype(str) + y3_test.astype(str) + y4_test.astype(str)+ y5_test.astype(str)


#resposta em HTTP
result =  pandas.concat([dataTest['NU_INSCRICAO'], y_test], axis=1, ignore_index=True)
result.columns = ['NU_INSCRICAO', 'TX_RESPOSTAS_MT']

answers = result.to_dict('records')
response = {'token':'efb1548e92ce3b7afba62f4d7f594ffd1e743ff7', 'email':'igor@gmail.com', 'answer':answers}

r = requests.post("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit", data = json.dumps(response))

