#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc import imshow
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image
import cv2
from PIL import Image, ImageOps
from scipy.misc import toimage, fromimage
from skimage.morphology import skeletonize
import re

###from skimage.feature import corner_harris, corner_subpix, corner_peaks

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


#########################################################################################

in_arg = sys.argv[1]
name = 'binary_saltpepper_invert_log_'+str(in_arg)+'.png'
a1 = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
a2 = a1.astype(float)
thresh = 180
######################################################################################

#perform median filter (get rid of specks or salt pepper noise)
#m = scipy.ndimage.filters.median_filter(b,size=2,footprint=None,output=None,mode='reflect',cval=0.0,origin=0)


row, col = a2.shape
#print(a2.shape)

for m in range (0, row):
    for n in range (0, col):
            Ixy = a2[m, n]
            #I = int(Ixy)
            if Ixy >= thresh:
                a2[m, n] = 255
            if Ixy < thresh:
                a2[m, n] = 0

#binary = toimage(a2)
#binary.save('binary_'+str(in_arg)+'.jpeg')

a3 = (a2)/np.max(a2)
a3 = (a3 == 1)
b = skeletonize(a3)
b = b.astype(np.float32)
c = toimage(b)
c.save('skeleton_'+str(in_arg)+'.png')

