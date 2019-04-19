# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:40:44 2018

@author: igor
"""

import requests
import pandas
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

fileTrain = pandas.read_csv('train.csv')
fileTest = pandas.read_csv('test4.csv')


x_train = fileTrain[['NU_IDADE', 'TP_ST_CONCLUSAO']] #, 'TP_ANO_CONCLUIU'
y_train = fileTrain[['IN_TREINEIRO']] #, 'TP_ANO_CONCLUIU'

x_test = fileTest[['NU_IDADE', 'TP_ST_CONCLUSAO']] #, 'TP_ANO_CONCLUIU'



sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)





cv = SVC(kernel = 'rbf', C = 200)
#cv = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
cv.fit(x_train, y_train)


#PREDICTION
pred = cv.predict(x_test)
pred_DF = pandas.DataFrame(data=pred)
#pred_DF[pred_DF<0] = 0


insc = fileTest[['NU_INSCRICAO']]
insc.reset_index(drop=True, inplace=True)

result =  pandas.concat([insc, pred_DF[0].apply(str)], axis=1, ignore_index=True)
result.columns = ['NU_INSCRICAO', 'IN_TREINEIRO']
answers = result.to_dict('records')
response = {'token':'efb1548e92ce3b7afba62f4d7f594ffd1e743ff7', 'email':'igor@gmail.com', 'answer':answers}



r = requests.post("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-4/submit", data = json.dumps(response))
r.text
