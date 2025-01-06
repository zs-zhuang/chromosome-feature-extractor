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

num_group = 2

#########################################################################################

def make_label(n):
    name = "group"+str(n)+"_feature_data"
    read = np.loadtxt(name, dtype=np.float32)
    out = "group"+str(n)+"_target"
    label_list = list()
    for i in range (0, len(read)):
        label_list.append(float(n))
    #np.savetxt(out, label_list, fmt='%1.0i')
    np.savetxt(out, label_list, fmt='%1.1f')



for x in range (1, num_group+1):
    make_label(x)







