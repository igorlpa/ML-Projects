# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:22:18 2019

@author: igorl
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler



class Aux:
    sex_label_encoder = LabelEncoder()
    embarked_label_encoder = LabelEncoder()
    hot_enconder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    sc_X = StandardScaler()

    def extractFeatures(self, X, isTraining=True):
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
        x = X[features]
    
        # tratamento de NaN
        x_nan = x.fillna(0)
        
        #add number of memeber families including the one 
        x2 = x_nan.copy()
        x2['family_number'] = x_nan.SibSp + x_nan.Parch + 1
        
        #categorical variables
        label_x = x2.copy()    
        if(isTraining):        
            label_x['Sex'] = self.sex_label_encoder.fit_transform(x2['Sex'].astype(str))
            label_x['Embarked'] = self.embarked_label_encoder.fit_transform(x2['Embarked'].astype(str))
            cols_ohe = pd.DataFrame( self.hot_enconder.fit_transform(label_x[['Sex', 'Embarked']] ))
        else:
            label_x['Sex'] = self.sex_label_encoder.transform(x2['Sex'].astype(str))
            label_x['Embarked'] = self.embarked_label_encoder.transform(x2['Embarked'].astype(str))
            cols_ohe = pd.DataFrame( self.hot_enconder.transform(label_x[['Sex', 'Embarked']] ))
        cols_ohe.index = label_x.index      
        x_OH = label_x.drop(['Sex', 'Embarked'], axis=1) 
        x_OH = pd.concat([x_OH, cols_ohe], axis=1)
        
        #standardScaler
        if(isTraining):
            x_final = self.sc_X.fit_transform(x_OH)
        else:
            x_final = self.sc_X.transform(x_OH)
            
        return x_final





# main
        
#features
train_file = pd.read_csv('train.csv')
teste_file = pd.read_csv('test.csv')

aux = Aux()
y_train = train_file[['Survived']]
x_train = aux.extractFeatures(train_file)

x_test = aux.extractFeatures(teste_file, False)
passengerID = pd.DataFrame(teste_file.PassengerId)


#classification


#SVM
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

steps = [ ('SVC', SVC())]
param_grid = {'SVC__C': [1, 10, 100, 200], 
             'SVC__kernel' : ['rbf', 'linear']}

estimator = Pipeline(steps=steps)
cv = GridSearchCV(estimator, param_grid, verbose=5, n_jobs=-1, cv=5)
cv.fit(x_train, y_train)
print(cv.best_params_, cv.best_score_)





pred = pd.DataFrame( cv.predict(x_test))
submission = pd.concat([passengerID, pred], axis=1)
submission.columns = ['PassengerId', 'Survived']

submission.to_csv('submission.csv', index=False)