#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc.pilutil import Image
from skimage import io
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from sklearn.preprocessing import OneHotEncoder

###from skimage.feature import corner_harris, corner_subpix, corner_peaks


#########################################################################################

# Import data 
featurefile = 'train_feature_data' #multi column file that contains feature values
targetfile = 'train_target_data' #single column file that specify whether a pixel is part of a good colony 1 or bad colony -1

X = np.loadtxt(featurefile)
Y = np.loadtxt(targetfile)

X = X.astype(np.float32)
Y = Y.astype(np.int32)

#print(X.shape, X.dtype)
#print(Y.shape, Y.dtype)

#Standardize or normalize features
#scaling features to lie between a given minimum and maximum value, often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size.
#The motivation to use this scaling include robustness to very small standard deviations of features and preserving zero entries in sparse data.

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#print(X.shape)
#print(X_scale.shape)
#print(X[:5])
#print(X_scale[:5])


newY = Y*2 -3

#reserve 30% of all data points for test set
from sklearn import model_selection as ms
X_train, X_test, Y_train, Y_test = ms.train_test_split(X_scale, newY, test_size=0.3)


#print(X_train.shape)

# Set up SVM
import cv2

#svm = cv2.ml.SVM_create()
#svm.setKernel(cv2.ml.SVM_Inter)
#svm.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)
#_, Y_pred = svm.predict(X_test)
#from sklearn import metrics
#accuracy = metrics.accuracy_score(Y_test, Y_pred)
#print(accuracy)

def train_svm(X_train, Y_train):
	svm = cv2.ml.SVM_create()
	svm.setKernel(cv2.ml.SVM_RBF) #try SVM_LINEAR, SVM_INTER, SVM_SIGMOID, SVM_RBF
	#svm.setC(1e-6)
	#svm.setGamma(1e-6)
	svm.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)
	return svm


def score_svm(svm, X, Y):
	from sklearn import metrics	
	_, Y_pred = svm.predict(X)
	return metrics.accuracy_score(Y,Y_pred)


def precision_svm(svm, X, Y):
        from sklearn import metrics
        _, Y_pred = svm.predict(X)
        return metrics.precision_score(Y,Y_pred)


def recall_svm(svm, X, Y):
        from sklearn import metrics
        _, Y_pred = svm.predict(X)
        return metrics.recall_score(Y,Y_pred)


svm = train_svm(X_train, Y_train)
score_train = score_svm(svm, X_train, Y_train)
score_test = score_svm(svm, X_test, Y_test)
score_train_precision = precision_svm(svm, X_train, Y_train)
score_test_precision = precision_svm(svm, X_test, Y_test)
score_train_recall = recall_svm(svm, X_train, Y_train)
score_test_recall = recall_svm(svm, X_test, Y_test)

print(score_train, score_test)
print(score_train_precision, score_test_precision)
print(score_train_recall, score_test_recall)


#########################################################

"""
# Testing and Debugging Only

test = np.random.random((10,5))
print(test.shape)
print(test)
print(test.dtype)
test32 = test.astype(np.float32) 
#_, testP =svm.predict(test)
print(test32.shape)
print(test32)
print(test32.dtype)


test2 = X_train[:10]
print(test2.shape)
print(test2)
print(test2.dtype)
_, testP =svm.predict(test32)


featurefile = 'test_feature_fix_stemcell1' #multi column file that contains feature values
test4 = np.loadtxt(featurefile)
test4 = test4.astype(np.float32)
print(test4.shape)
print(test4.dtype)
_, testP =svm.predict(test4)
"""

#########################################################
