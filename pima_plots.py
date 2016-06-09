# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:37:14 2016

@author: XPS
"""
import pandas as pd
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pimadiabetes.csv', skiprows=1, names=names)
""""
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
print(data.describe())

## plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#data.boxplot()
#data.hist()
#data.groupby('class').hist()
#data.groupby('class').plas.hist(alpha=0.4)
from pandas.tools.plotting import scatter_matrix
#scatter_matrix(data, alpha=0.2, figsize=(16.0, 16.0), diagonal='kde')
#plt.savefig(r"scatter_matrix_pima.png")

# Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(dataset.data, dataset.target)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
