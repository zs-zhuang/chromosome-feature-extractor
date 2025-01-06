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
name = in_arg+'.bmp'
img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)


b = cv2.equalizeHist(img)
b2 = scipy.misc.toimage(b)
b2.save('hist_'+str(in_arg)+'.png')


log1 = img.astype(float)
log2 = np.max(log1)
log3 = (255.0*np.log(1+log1))/np.log(1+log2)
log4 = log3.astype(int)
log5 = scipy.misc.toimage(log4)
log5.save('log_'+str(in_arg)+'.png')


#r = 5
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(r,r))
#cl1 = clahe.apply(img)
#cl2 = scipy.misc.toimage(cl1)
#cl2.save('clahe'+str(r)+'_'+str(in_arg)+'.png')
