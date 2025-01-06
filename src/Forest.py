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
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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


#create tree
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features=20, max_depth=4)

forest.fit(X_train, Y_train)
score_train = forest.score(X_train, Y_train)
score_test = forest.score(X_test, Y_test)

print(score_train, score_test)




