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
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pimadiabetes.csv', skiprows=1, names=names)
# making a target column for the thing we want to predict
target = data['class']
# removing it from the data (otherwise that would be circular)
del data['class'] 

X = data
y = target
target_names = ['no diabetes','diabetes']

# splitting data into train and test; using 80/20 split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
 
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
for c, i, target_name in zip("rb", [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')

plt.figure()
for c, i, target_name in zip("rb", [0, 1], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.show()