#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc import imshow
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image

from PIL import Image, ImageOps
import cv2
###from skimage.feature import corner_harris, corner_subpix, corner_peaks


#########################################################################################

in_arg = sys.argv[1]
name = 'invert_log_'+str(in_arg)+'.png'
img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)


#permform mean filter (blur image, reduce resolution which could mean reduce noise)
#initialize mean filter size 2 by 2 (higher filter size causes more blur image)
#r = 5
#k = np.ones((r,r))/(r*r)

# perform convolution
#b = scipy.ndimage.filters.convolve(img,k)

#perform median filter (get rid of specks or salt pepper noise)
c = scipy.ndimage.filters.median_filter(img,size=3,footprint=None,output=None,mode='reflect',cval=0.0,origin=0)

#b2 = scipy.misc.toimage(b)
#b2.save('denoise_'+str(in_arg)+'.jpeg')

c2 = scipy.misc.toimage(c)
#c2.save('denoise'+str(r)+'_'+str(in_arg)+'.jpeg')
c2.save('saltpepper_invert_log_'+str(in_arg)+'.png')
