#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc import imshow
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image

from PIL import Image, ImageOps
import cv2
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

#########################################################################################
#in_arg = sys.argv[1]

myfile = np.loadtxt("all_feat", dtype=np.float32)
r, c, = myfile.shape
#print(r,c)

outlier_location = set()

for f in range (0, c-1):
#for f in range(0, 1):
    x = myfile[:,f]
    mean = np.mean(x)
    std = np.std(x)
    upperbound = mean+3*std
    lowerbound = mean-3*std
    #print(upperbound, lowerbound)
    #print(f, mean, std)
    for i in range (0, r):
        if x[i] >= upperbound or x[i] < lowerbound:
            outlier_location.add(i)


#print(outlier_location)

newlist = list()
for h in range(0, r):
    if h not in outlier_location:
        newlist.append(myfile[h])

print(len(myfile), len(newlist))
np.savetxt("clean_all_feat", newlist)
