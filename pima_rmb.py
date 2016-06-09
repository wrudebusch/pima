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
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

###############################################################################
# Reading data

# naming the coulmns
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
X = pd.read_csv('pimadiabetes.csv', skiprows=1, names=names)

# making a target column for the thing we want to predict
y = X['class']
# removing it from the data (otherwise that would be circular)
del X['class']

# splitting data into train and test; using 80/20 split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)

###############################################################################
# Setting up models

# attempting to do a Bernoulli Restricted Boltzmann Machine
# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
binarizer = preprocessing.Binarizer(threshold=1.1)
#X = binarizer.transform(X)

classifier = Pipeline(steps=[('binarizer', binarizer), ('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

###############################################################################