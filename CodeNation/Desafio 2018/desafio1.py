# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import requests
import pandas
import numpy as np
import json

file = pandas.read_csv('train.csv')

data = file[['NU_INSCRICAO', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']]

resultado = 2*data.values[:,1] + 1*data.values[:,2] + 1.5*data.values[:,3] + 3*data.values[:,4] + 3*data.values[:,5]
resultado = resultado/(2+1+1.5+3+3)

inscricoes = data.values[:,0]

r = np.vstack((inscricoes,resultado)).T

df = pandas.DataFrame(data=r)

sortedDF = df.sort_values(1,  ascending=[False])[:20].round(1)
sortedDF.columns = ['NU_INSCRICAO', 'NOTA_FINAL']

answers = sortedDF.to_dict('records')

response = {'token':'efb1548e92ce3b7afba62f4d7f594ffd1e743ff7', 'email':'igor@gmail.com', 'answer':answers}

r = requests.post("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-1/submit", data = json.dumps(response))
