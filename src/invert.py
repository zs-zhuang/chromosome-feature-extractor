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
name = 'log_'+str(in_arg)+'.png'
img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

invert = cv2.bitwise_not(img)


i = scipy.misc.toimage(invert)
i.save('invert_log_'+str(in_arg)+'.png')
