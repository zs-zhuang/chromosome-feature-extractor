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
in_arg = sys.argv[1]
x = np.loadtxt(in_arg, dtype=np.float32)

#x = myfile[:,0]

mean = np.mean(x)
median = np.median(x)

std = np.std(x)
max_x = np.max(x)
min_x = np.min(x)

print("mean: ", mean)
#print("median:", median)
print("std:", std)
print("max:", max_x)
print("min:", min_x)
