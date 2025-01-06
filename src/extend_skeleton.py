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
# get skeleton file and derivative file 
in_arg = sys.argv[1] #name of file containing skeleton positions
in_name = str(in_arg) + '.txt2'
in_name2 = str(in_arg) + '.der_poly3'
skel = np.loadtxt(in_name, dtype=int)
der = np.loadtxt(in_name2, dtype=np.float32)
der_x = der[:, 0]
der_y = der[:, 1]
i = np.arange(0, len(der_x), 1)
l = len(skel)
#l, c = skel.shape

#read in binary karyotyping, get all pixel position that belongs to chromosome into d_white
#img = "binary_clahe5_blur_saltpepper_clean_Normal_Female_Key1_ZWK99004k.jpeg"

img = "binary_saltpepper_invert_log_" + str(in_arg)+".png"

a = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
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


s_skel = set([])
for i in range (0, l):
    x = skel[i][0]
    y = skel[i][1]
    row = (x, y)
    if row not in s_skel:
        s_skel.add(row)
#print(len(skel), len(s_skel))


def extend_skel(H):
    if H in s_white:
        r = H[0]
        c = H[1]
        e = (r,c)
        top_skel_list.append(e)


def extend_skel2(H):
    if H in s_white:
        r = H[0]
        c = H[1]
        e = (r,c)
        bot_skel_list.append(e)



top_skel_list = list()
top_skel_list2 = list()
top_der_list = list()
p0 = skel[0][0]
q0 = skel[0][1]
pos0 = (p0, q0)
sx0 = der_x[0]
sy0 = der_y[0]
der0 = (sx0, sy0)
T1 = (int(p0-sx0), int(q0-sy0))
T2 = (int(p0-2*sx0), int(q0-2*sy0))
T3 = (int(p0-3*sx0), int(q0-3*sy0))
T4 = (int(p0-4*sx0), int(q0-4*sy0))
T5 = (int(p0-5*sx0), int(q0-5*sy0))
T6 = (int(p0-6*sx0), int(q0-6*sy0))
T7 = (int(p0-7*sx0), int(q0-7*sy0))
T8 = (int(p0-8*sx0), int(q0-8*sy0))
T9 = (int(p0-9*sx0), int(q0-9*sy0))
T10 = (int(p0-10*sx0), int(q0-10*sy0))
#print(pos0)
#print(sx0, sy0)
#print(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)
extend_skel(T10)
extend_skel(T9)
extend_skel(T8)
extend_skel(T7)
extend_skel(T6)
extend_skel(T5)
extend_skel(T4)
extend_skel(T3)
extend_skel(T2)
extend_skel(T1)
#print(len(top_skel_list))
#print(top_skel_list)

for t in range(0, len(top_skel_list)):
        row = top_skel_list[t]
        if row not in s_skel:
            s_skel.add(row)
            top_skel_list2.append(row)
            top_der_list.append(der0)

#print(len(top_skel_list), len(top_skel_list2))

bot_skel_list = list()
bot_skel_list2 = list()
bot_der_list = list()
pl = skel[l-1][0]
ql = skel[l-1][1]
posl = (pl, ql)
sxl = der_x[l-1]
syl = der_y[l-1]
derl = (sxl, syl)
B1 = (int(pl+sxl), int(ql+syl))
B2 = (int(pl+2*sxl), int(ql+2*syl))
B3 = (int(pl+3*sxl), int(ql+3*syl))
B4 = (int(pl+4*sxl), int(ql+4*syl))
B5 = (int(pl+5*sxl), int(ql+5*syl))
B6 = (int(pl+6*sxl), int(ql+6*syl))
B7 = (int(pl+7*sxl), int(ql+7*syl))
B8 = (int(pl+8*sxl), int(ql+8*syl))
B9 = (int(pl+9*sxl), int(ql+9*syl))
B10 = (int(pl+10*sxl), int(ql+10*syl))
#print(posl)
#print(sxl, syl)
#print(B1, B2, B3, B4, B5, B6, B7, B8, B9, B10)
extend_skel2(B1)
extend_skel2(B2)
extend_skel2(B3)
extend_skel2(B4)
extend_skel2(B5)
extend_skel2(B6)
extend_skel2(B7)
extend_skel2(B8)
extend_skel2(B9)
extend_skel2(B10)
#print(len(bot_skel_list))
#print(bot_skel_list)

for b in range(0, len(bot_skel_list)):
        row = bot_skel_list[b]
        if row not in s_skel:
            s_skel.add(row)
            bot_skel_list2.append(row)
            bot_der_list.append(derl)

#print(len(bot_skel_list), len(bot_skel_list2))


skel_list = np.ndarray.tolist(skel)
der_list = np.ndarray.tolist(der)
final_skel_list = top_skel_list2 + skel_list + bot_skel_list2
final_der_list = top_der_list + der_list + bot_der_list

#print(len(skel_list), len(top_skel_list2), len(bot_skel_list2), len(final_skel_list))
#print(final_skel_list)


final_skel = np.asarray(final_skel_list)
final_der = np.asarray(final_der_list)
final_size = len(final_skel_list)

outname = str(in_arg)+".txt3"
outname2 = str(in_arg)+".der_poly3_ext"

np.savetxt(outname, final_skel, fmt='%1.0i')
np.savetxt(outname2, final_der)

########################################################################################

