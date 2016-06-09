# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:37:14 2016

@author: Will Rudebusch
"""
#from sklearn import datasets
#iris = datasets.load_iris()
#y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
"""
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
import numpy as np
from sklearn.cross_validation import train_test_split

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pimadiabetes.csv', skiprows=1, names=names)
target = data['class'] #making a target column for the thing we want to predict
del data['class'] #removing it from the data (otherwise that would be ciruclar)
train, test = train_test_split(data, test_size = 0.2) # splitting data into train and test

print data.describe()
print '\n shape:', data.shape

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data, target).predict(data)

print("Number of mislabeled points out of a total %d points : %d"
      % (data.shape[0],(target != y_pred).sum()))
