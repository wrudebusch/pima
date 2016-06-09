# -*- coding: utf-8 -*-
'''Pima Indians Diabetes Data
   (https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
   1. preg = Number of times pregnant
   2. plas = Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. pres = Diastolic blood pressure (mm Hg)
   4. skin = Triceps skin fold thickness (mm)
   5. test = 2-Hour serum insulin (mu U/ml)
   6. mass = Body mass index (weight in kg/(height in m)^2)
   7. pedi = Diabetes pedigree function
   8. age = Age (years)
   9. class = diabetes within 5yrs of the measurements (1) or not (0)
'''

import pandas as pd
from sklearn.cross_validation import train_test_split

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pimadiabetes.csv', skiprows=1, names=names)
# making a target column for the thing we want to predict
target = data['class']
# removing it from the data (otherwise that would be circular)
del data['class'] 

X = data
y = target

# splitting data into train and test; using 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
 
# using the standard SVM model
from sklearn import svm
clf = svm.SVC(gamma=0.0001, C=1000.)
y_predict = clf.fit(X_train, y_train).predict(X_test)

# printing the success rate
print("success rate",sum(y_test-y_predict==0)/float(len(y_test)))