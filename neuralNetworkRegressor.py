import numpy as np
import math

class nnr:

  def __init__(self, width, depth, activation='relu', iterations=10000, learnRate=0.33):
    self.width = width
    self.depth = depth
    self.activation = activation
    self.iterations = iterations
    self.learnRate = learnRate

  def relu(self, x):
    return np.maximum(x, 0)

  def sigmoid(self, x):
    return (1 / (1 + np.exp(-x)))

  def initWeightsAndBias(self, X):
    # Weight initialization based on the technique found at https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

    self.weights0 = np.random.randn(X.shape[1], self.width) * np.sqrt(2 / X.shape[1])
    self.bias0 = np.random.randn(1, self.width)

    self.weights = np.random.randn(self.depth, self.width, self.width) * np.sqrt(2 / self.width)
    self.bias = np.random.randn(self.depth, 1, self.width)

    self.weightsn = np.random.randn(self.width, 1) * np.sqrt(2 / self.width)
    self.biasn = np.random.randn(1)

  def mse(self):
    self.cost = 1 / self.finalValues.shape[0] * np.sum(np.square(self.finalValues - self.Y))

  def r2(self, Y):
    self.score = 1 - (np.sum(np.square(self.finalValues - Y)) / np.sum(np.square(Y - np.mean(Y))))

  def forward(self, X):
    self.values0 = np.dot(X, self.weights0) + self.bias0
    if self.activation == 'relu':
      self.values0 = self.relu(self.values0)
    elif self.activation == 'sigmoid':
      self.values0 = self.sigmoid(self.values0)

    self.values = np.zeros((self.depth, X.shape[0], self.width))

    for i in range(0, self.depth):
      if i == 0:
        self.values[0] = np.dot(self.values0, self.weights[0]) + self.bias[0]
      else:
        self.values[i] = np.dot(self.values[i-1], self.weights[i]) + self.bias[i]
      
      if self.activation == 'relu':
        self.values[i] = self.relu(self.values[i])
      elif self.activation == 'sigmoid':
        self.values[i] = self.sigmoid(self.values[i])

    self.finalValues = np.dot(self.values[self.depth-1], self.weightsn) + self.biasn

  def backward(self):
    length = self.finalValues.shape[0]

    dLayer = 2 / length * (self.finalValues - self.Y)

    self.d_weightsn = 2 / length * np.dot(dLayer.T, self.values[self.depth-1]).T
    self.d_biasn = 2 / length * np.sum(dLayer)

    d_weights = np.zeros((self.depth, self.width, self.width))
    d_bias = np.zeros((self.depth, 1, self.width))

    for i in range(self.depth-1, -1, -1):
      if i == self.depth-1:
        dLayer = np.dot(self.weightsn, dLayer.T).T
      else:
        dLayer = np.dot(self.weights[i+1], dLayer.T).T

      if self.activation == 'relu':
        dLayer = dLayer * np.where(self.values[i] >= 0, 1, 0)
      elif self.activation == 'sigmoid':
        dLayer = dLayer * (self.sigmoid(self.values[i]) * (1 - self.sigmoid(self.values[i])))

      if i == 0:
        d_weights[0] = 2 / length * np.dot(dLayer.T, self.values0).T
      else:
        d_weights[i] = 2 / length * np.dot(dLayer.T, self.values[i-1]).T
      d_bias[i] = 2 / length * np.sum(dLayer)

    dLayer = np.dot(self.weights[0], dLayer.T).T
    if self.activation == 'relu':
      dLayer = dLayer * np.where(self.values0 >= 0, 1, 0)
    elif self.activation == 'sigmoid':
      dLayer = dLayer * (self.sigmoid(self.values0) * (1 - self.sigmoid(self.values0)))

    d_weights0 = 2 / length * np.dot(dLayer.T, self.X).T
    d_bias0 = 2 / length * np.sum(dLayer)

    self.weights0 = self.weights0 - self.learnRate * d_weights0
    self.bias0 = self.bias0 - self.learnRate * d_bias0

    for i in range(0, self.depth):
     self.weights[i] = self.weights[i] - self.learnRate * d_weights[i]
     self.bias[i] = self.bias[i] - self.learnRate * d_bias[i]

    self.weightsn = self.weightsn - self.learnRate * self.d_weightsn
    self.biasn = self.biasn - self.learnRate * self.d_biasn

  def fit(self, X, Y):
    self.X = X
    self.Y = Y

    self.initWeightsAndBias(X)

    for i in range(0, self.iterations):
      self.forward(self.X)
      self.mse()
      #print('cost at iteration ' + str(i) + ': ' + str(self.cost))
      if np.isnan(self.cost):
        self.initWeightsAndBias(X)
        self.learnRate *= 0.5
        continue
      self.backward()

    self.r2(self.Y)
    return self.score
    

  def predict(self, X):
    self.forward(X)
    return self.finalValues

  def kFoldCrossValidation(self, X, Y, k=10):
    np.random.shuffle(X)
    np.random.shuffle(Y)

    Xs = np.array_split(X, 10)
    Ys = np.array_split(Y, 10)

    scores = []

    for i in range(0, k):
      if i == 0:
        XTrain = np.concatenate(Xs[i+1:])
        YTrain = np.concatenate(Ys[i+1:])
      elif i == k-1:
        XTrain = np.concatenate(Xs[:i])
        YTrain = np.concatenate(Ys[:i])
      else:
        XTrain = np.vstack((np.concatenate(Xs[:i]), np.concatenate(Xs[i+1:])))
        YTrain = np.vstack((np.concatenate(Ys[:i]), np.concatenate(Ys[i+1:])))

      XTest = Xs[i]
      YTest = Ys[i]

      self.fit(XTrain, YTrain)
      YPred = self.predict(XTest)
      self.r2(YTest)
      scores.append(self.score)

    return np.mean(scores)

