# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:32:54 2018

@author: igorl
"""

import requests
import pandas
import numpy as np
import json
import matplotlib.pyplot as plt


fileTrain = pandas.read_csv('train.csv')
fileTest = pandas.read_csv('test3.csv')


x_train = getX(fileTrain, True)

x_test = getX(fileTest, False)


def getX( file, isTrain):
   data = file[['NU_INSCRICAO', 'TX_RESPOSTAS_MT', 'CO_PROVA_MT']].dropna()
   s = pandas.Series(data.iloc[:,1])
   if(isTrain):
       q = s.str.extract('([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])', expand=True)   
   else:
       q = s.str.extract('([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])([A-Z.*])', expand=True)
   qReplaced = q.replace(['A', 'B', 'C', 'D', 'E', '.', '*'], [1,2,3,4,5, 6,6])     
   X = pandas.concat([data['CO_PROVA_MT'], qReplaced.iloc[:, :40]], axis=1, ignore_index=True)    
   return X