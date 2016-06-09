# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:37:14 2016

@author: Will Rudebusch

data source: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
1. preg = Number of times pregnant
2. plas = Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. pres = Diastolic blood pressure (mm Hg)
4. skin = Triceps skin fold thickness (mm)
5. test = 2-Hour serum insulin (mu U/ml)
6. mass = Body mass index (weight in kg/(height in m)^2)
7. pedi = Diabetes pedigree function
8. age = Age (years)
9. class = diabetes within 5yrs of the measurements (1) or not (0)
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
# naming the coulmns
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
X = pd.read_csv('pimadiabetes.csv', skiprows=1, names=names)
# making a target column for the thing we want to predict
y = X['class']
# removing it from the data (otherwise that would be circular)
del X['class'] 
# splitting data into train and test; using 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
 
# using the standard SVM model
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
# this command split the data and traning set 3 times and tests it
predicted = cross_val_predict(clf, X, y, cv=3)
print accuracy_score(y, predicted)