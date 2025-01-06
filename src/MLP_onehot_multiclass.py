#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc.pilutil import Image
from skimage import io
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
import cv2
from sklearn import model_selection as ms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
###from skimage.feature import corner_harris, corner_subpix, corner_peaks


#########################################################################################

#in_arg = sys.argv[1]
#common_name = in_arg

# Import data 
featurefile = 'train_feature_data' #multi column file that contains feature values
targetfile = 'train_target_data' #single column file that specify whether a pixel is part of a good colony 1 or bad colony -1

X = np.loadtxt(featurefile)
Y = np.loadtxt(targetfile)

X = X.astype(np.float32)
Y_original = Y.astype(np.float32)

#print(X.shape, X.dtype)
#print(Y.shape, Y.dtype)

#Standardize or normalize features

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

print(X_scale.shape)

#one-hot encoding target label
enc = OneHotEncoder(sparse=False, dtype=np.float32)
Y_onehot = enc.fit_transform(Y_original.reshape(-1, 1))
print(Y_onehot.shape)


#reserve 20% of all data points for test set
X_train, X_test, Y_train, Y_test = ms.train_test_split(X_scale, Y_onehot, test_size=0.3)


#set up MLP, n hidden layer
mlp = cv2.ml.ANN_MLP_create()
n_input = 42
n_hidden1 = 24
#n_hidden2 = 36
n_output = 2
mlp.setLayerSizes(np.array([n_input, n_hidden1, n_output]))
mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2.5, 1.0)
mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
mlp.setBackpropWeightScale(0.0001)
term_mode = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
term_max_iter = 1000
term_eps = 0.001
mlp.setTermCriteria((term_mode, term_max_iter, term_eps))

#run mlp
#mlp.train(X_scale, cv2.ml.ROW_SAMPLE, Y_onehot)
#_, Y_pred = mlp.predict(X_scale)
#Y_pred2 = np.sign(Y_pred)

mlp.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)
_, Y_pred_train = mlp.predict(X_train)
score_train = accuracy_score(Y_pred_train.round(), Y_train)

_, Y_pred_test = mlp.predict(X_test)
score_test = accuracy_score(Y_pred_test.round(), Y_test)


#np.savetxt('Y_onehot_training', Y_onehot, fmt='%1.0i')
#np.savetxt('Y_pred_training', Y_pred.round(), fmt='%1.0i')
#np.savetxt('Y_pred_training_raw', Y_pred)

#score_scikit = accuracy_score(Y_pred.round(), Y_onehot)
print(score_train, score_test)

##########################################################################

#check accuracy score by hand, same result as score_scikit above
#raw_score = 0
#for s in range (0, len(Y_onehot)):
#    true_label = Y_onehot[s]
#    pred_label = Y_pred[s].round()
    #print(true_label, pred_label)
#    test = (true_label==pred_label).all()
#    test2 = str(test)
    #print(test2) #if same, will print true, else will print false
#    if test2 == "False":
#        raw_score = raw_score + 1

#print(raw_score, len(Y_onehot))
#N_correct = len(Y_onehot) - raw_score
#score = N_correct/len(Y_onehot)
#print(score)

##########################################################################

"""
# Make Prediction
featurefile = ('feature_'+common_name) #multi column file that contains feature values
predictionfile = ('prediction_'+common_name)

test = np.loadtxt(featurefile)
test = test.astype(np.float32)
test_scale = min_max_scaler.fit_transform(test)

_, testP =mlp.predict(test_scale)
testP2 = np.sign(testP)
#testP3 = testP.astype(int)
np.savetxt(predictionfile, testP2, fmt='%1.0i')
"""
