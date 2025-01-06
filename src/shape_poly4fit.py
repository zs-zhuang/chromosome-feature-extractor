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
in_name = str(in_arg) + '.shape'
coor = np.loadtxt(in_name, dtype=np.int)

x = coor[:, 0]
y = coor[:, 1]
i = np.arange(0, len(x), 1)
#print(x)
#print(i)
l = len(x)
#print(l)

#fit x and y independently to i using 3rd order polynomial
order = int(len(x)/10)
z = np.polyfit(x, y, order)
p = np.poly1d(z)
#zx = np.polyfit(i, x, order)
#px = np.poly1d(zx)
#zy = np.polyfit(i, y, order)
#py = np.poly1d(zy)

#get predicted y value (ynew) from the fitted curve for given x
ynew = p(x)
fit = np.vstack((x, ynew)).T

#get derivative of the curve
#d = np.polyder(p)
#dx = d(x)
#der = np.vstack((x, dx)).T
#dx = np.polyder(px)
#d_ix = dx(i)
#dy = np.polyder(py)
#d_iy = dy(i)

#der_ixy = np.vstack((d_ix, d_iy)).T


#outname = "C30_size_90.fit_poly"+str(order)
#outname2 = "C30_size_90.der_poly"+str(order)
#outname = str(in_arg)+".fit_poly"+str(order)
outname2 = str(in_arg)+".shape2"
#np.savetxt(outname, fit)
np.savetxt(outname2, fit)
#np.savetxt(outname, fit, fmt='%1.0i')


########################################################################################

########################################################################################

#####spline fit, does not work in this case
#x_max = np.amax(x)
#x_min = np.amin(x)
#l = len(x)
#xnew = np.arange(x_min, x_max+0.2, 0.2)
#tck = interpolate.splrep(x, y, s=0)
#ynew = interpolate.splev(xnew, tck, der=0)
#new = np.vstack((xnew, ynew)).T
#outname = "test_spline"
#np.savetxt(outname, new)

#####1d interpolation interp1d, does not work in this case
#f = interp1d(x, y, kind = 'cubic')
#x_max = np.amax(x)
#x_min = np.amin(x)
#l = len(x)
#xnew = np.linspace(x_min, x_max, num = l*5, endpoint=True)
#ynew = f(xnew)
#new = np.vstack((xnew, ynew)).T
#outname = "test_interp1d"
#np.savetxt(outname, new)
