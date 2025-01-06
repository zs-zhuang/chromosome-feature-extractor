#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc import imshow
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image
import scipy.misc

from PIL import Image, ImageOps
import cv2
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

#########################################################################################
# get skeleton file and derivative file 
in_arg = sys.argv[1] #original karyotyping image name without extension
in_name = str(in_arg) + '.txt3'
in_name2 = str(in_arg) + '.der_poly3_ext'
skel = np.loadtxt(in_name, dtype=int)
l, c = skel.shape
der = np.loadtxt(in_name2, dtype=np.float32)
der_x = der[:, 0]
der_y = der[:, 1]
#i = np.arange(0, len(der_x), 1)


#read in original karyotyping image and its binary version, get all pixel position that belongs to chromosome into d_white
img = str(in_arg) + '.bmp'
img2 = 'binary_saltpepper_invert_log_'+str(in_arg) +'.png'
o = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
a = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
(thresh, b) = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
height, width = b.shape
s_white = set([])
for m in range (0, height):
    for n in range (0, width):
        if b[m][n] == 255:
            location = (m, n)
            if location not in s_white:
                s_white.add(location)

#print(len(s_white))

#create a new image/array with the same size as original kayotype image, solid black blackground, convert perpendicular band to the skeleton to white pixels in order to visualize the bands are extracted correctly
#verify_band = np.zeros(b.shape)



#get average pixel brightness of the skeleton
skel_brightness = list()
for j in range(0, l):
    s = skel[j][0]
    t = skel[j][1]
    skel_brightness.append(o[s][t])

Arr_skel_brightness = np.asarray(skel_brightness)
avg_skel_brightness = np.mean(Arr_skel_brightness)
#print(l, len(skel_brightness))
#print(skel_brightness)
#print(avg_skel_brightness)


#grab pixels belonging to a given band and calculate average
#grab +7 and -7 pixels from each side of the skeleton, only keep the ones in s_white
def gather_band_pix(H):
    if H in s_white:
        r = H[0]
        c = H[1]
        band.append(o[r][c])
        #verify_band[r][c] = 255

#prepare output file
outname = str(in_arg)+".band"
outname2 = str(in_arg)+".shape"
#outname3 = str(in_arg)+".std"
outfile = open (outname, 'w')
outfile2 = open (outname2, 'w')
#outfile3 = open (outname3, 'w')
#print(l)



# get band brightness for each position along the length of skeleton
for i in range(0, l):

    band = list()

    p = skel[i][0]
    q = skel[i][1]
    pos = (p, q)
    sx = der_x[i]
    sy = der_y[i]
    dx = sy
    dy = -sx
    #print(pos)
    #print(dx, dy)

    RH1 = (int(p+dx), int(q+dy)) 
    RH2 = (int(p+2*dx), int(q+2*dy))
    RH3 = (int(p+3*dx), int(q+3*dy))
    RH4 = (int(p+4*dx), int(q+4*dy))
    RH5 = (int(p+5*dx), int(q+5*dy))
    RH6 = (int(p+6*dx), int(q+6*dy))
    RH7 = (int(p+7*dx), int(q+7*dy))
    RH8 = (int(p+8*dx), int(q+8*dy))
    RH9 = (int(p+9*dx), int(q+9*dy))
    RH10 = (int(p+10*dx), int(q+10*dy))

    LH1 = (int(p-dx), int(q-dy))
    LH2 = (int(p-2*dx), int(q-2*dy))
    LH3 = (int(p-3*dx), int(q-3*dy))
    LH4 = (int(p-4*dx), int(q-4*dy))
    LH5 = (int(p-5*dx), int(q-5*dy))
    LH6 = (int(p-6*dx), int(q-6*dy))
    LH7 = (int(p-7*dx), int(q-7*dy))
    LH8 = (int(p-8*dx), int(q-8*dy))
    LH9 = (int(p-9*dx), int(q-9*dy))
    LH10 = (int(p-10*dx), int(q-10*dy))
    
    #print(RH1, RH2, RH3, RH4, RH5, RH6, RH7, LH1, LH2, LH3, LH4, LH5, LH6, LH7)
    gather_band_pix(pos)
    gather_band_pix(RH1)
    gather_band_pix(RH2)
    gather_band_pix(RH3)
    gather_band_pix(RH4)
    gather_band_pix(RH5)
    gather_band_pix(RH6)
    gather_band_pix(RH7)
    gather_band_pix(RH8)
    gather_band_pix(RH9)
    gather_band_pix(RH10)

    gather_band_pix(LH1)
    gather_band_pix(LH2)
    gather_band_pix(LH3)
    gather_band_pix(LH4)
    gather_band_pix(LH5)
    gather_band_pix(LH6)
    gather_band_pix(LH7)
    gather_band_pix(LH8)
    gather_band_pix(LH9)
    gather_band_pix(LH10)
    
    #print(band)
    Arr_band = np.asarray(band)
    band_mean = np.mean(Arr_band)
    band_mean_norm = band_mean/avg_skel_brightness
    band_width = len(Arr_band)
    #print(band_width)
    band_width_norm = band_width/l
    #i_norm = i/(l-1)
    band_std = np.std(Arr_band)
    #print(mean)
    outfile.write(str(i)+' '+str(band_mean_norm)+'\n')
    outfile2.write(str(i)+' '+str(band_width)+'\n')
    #outfile3.write(str(i)+' '+str(band_std)+'\n')


#verify_name = "verify_band_"+str(in_arg2)+".jpeg"
#scipy.misc.imsave(verify_name, verify_band)



########################################################################################

