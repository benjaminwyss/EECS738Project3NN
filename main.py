import sys
import numpy as np
import neuralNetworkRegressor as nnr

import warnings
warnings.filterwarnings("ignore")

X = np.loadtxt('data/housingCleaned.csv', delimiter=',')
Y = X[:, -1:]
X = X[:, :-1]

model = nnr.nnr(13, 3, activation='relu', iterations=10000, learnRate=0.001)
print('Housing Dataset:')
print('training r2 score: ' + str(model.fit(X, Y)))
print('10-fold cross validation r2 score: ' + str(model.kFoldCrossValidation(X, Y)))

X = np.loadtxt('data/bikesCleaned.csv', delimiter=',')
Y = X[:, -1:]
X = X[:, :-1]

model = nnr.nnr(12, 4, activation='relu', iterations=1000, learnRate=0.01)
print('\nBikes Dataset:')
print('training r2 score: ' + str(model.fit(X, Y)))
print('10-fold cross validation r2 score: ' + str(model.kFoldCrossValidation(X, Y)))