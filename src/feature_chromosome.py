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
tag = sys.argv[2]
terms = 10 #number of fft terms to keep as features

in_arg = sys.argv[1]
band_name = str(in_arg) + '.band'
shape_name = str(in_arg) + '.shape'
shape2_name = str(in_arg) + '.shape2'
skel_name = str(in_arg) + '.txt3'

band = np.loadtxt(band_name, dtype=np.float32)
shape = np.loadtxt(shape_name, dtype=np.float32)
shape2 = np.loadtxt(shape2_name, dtype=np.float32)
skel = np.loadtxt(skel_name, dtype=int)

B = band[:,1]
S = shape[:,1]
S2 = shape2[:,1]
x = band[:,0]

out_name = str(in_arg)+'.feat'
feat_list = list() 

#########################################################################################

#1.calculate length to width ratio, width = avg width from shape profile
length = len(skel)
if length < terms:
    print("chromosome is too short")
    sys.exit(1)

width = np.mean(S)
R = length/width
feat_list.append(R)

#2.calculate location of centromere, measured in percentage of total length from the tip of shorter arm
ends = 0.3
start = int(ends*length)
end = int((1-ends)*length)
S2_mid = S2[start:end]
all_C_location = (np.where(S2_mid==S2_mid.min()))
C_location = np.median(all_C_location)

original_L = len(S2)
short_L = len(S2_mid)
trim = int(ends*length)

true_C_location = C_location+trim+1

position1 = true_C_location/original_L
position2 = (original_L-true_C_location)/original_L

#if  position1 >=0.4 and position1 <=0.6:
#    centro_class = 0.0
#    feat_list.append(centro_class)
#else:
#    centro_class = 1.0
#    feat_list.append(centro_class)
#print(position1, centro_class)

if position1 <= position2:
    feat_list.append(position1)
    Cent = position1
else:
    #print("chromosome is inverted")
    feat_list.append(position2)
    Cent = position2
    #S = np.flip(S, 0)
    #B = np.flip(B, 0)
print(Cent)
#########################################################################################

#process band brightness curve

#3. calculat mean and std of skeleton brightness

cut = int(0.3*len(B))
N_cut = len(B) - cut
#C = int(Cent*original_L)
#print(len(B), cut, C)
#print(cut30, R_cut)

B_mid = B[cut:N_cut] 
x_mid = x[cut:N_cut]
#Bshort = B[0:C]
#Blong = B[C:]

feat_list.append(np.mean(B))
feat_list.append(np.mean(B_mid))
#feat_list.append(np.mean(Bshort))
#feat_list.append(np.mean(Blong))
feat_list.append(np.std(B))
feat_list.append(np.std(B_mid))
diff_bright = np.max(B) - np.min(B)
feat_list.append(diff_bright)
#feat_list.append(np.std(Bshort))
#feat_list.append(np.std(Blong))
#feat_list.append(np.max(B))
#feat_list.append(np.min(B))
#feat_list.append(np.max(B_mid))
#feat_list.append(np.min(B_mid))
#feat_list.append(np.max(Bshort))
#feat_list.append(np.min(Bshort))
#feat_list.append(np.max(Blong))
#feat_list.append(np.min(Blong))

#print(np.mean(B_mid), np.mean(B30), np.mean(BN30))
#print(np.std(B_mid), np.std(B30), np.std(BN30))
#print(np.max(B_mid), np.min(B_mid), np.max(B30), np.min(B30), np.max(BN30), np.min(BN30))


#4. calculate auto-correlation of brightness curve
def auto_corr(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

autoB = auto_corr(B)
feat_list.append(np.mean(autoB))
#feat_list.append(np.median(autoB))
feat_list.append(np.max(autoB))
#feat_list.append(np.min(autoB))
feat_list.append(np.std(autoB))

#print(np.mean(autoB), np.median(autoB), np.max(autoB), np.min(autoB), np.std(autoB))


#5. calculate fourier transform of brightness curve
w = np.fft.fft(B)
w2 = np.absolute(w)

#freqs = np.fft.fftfreq(len(w))
#feat_list.append(freqs.max())
#feat_list.append(freqs.min())
#print(w2[0:4])

for t in range (0, terms):
    feat_list.append(w2[t])
feat_list.append(np.mean(w2))
feat_list.append(np.std(w2))

#feat_list.append(w2[0])
#print(w2.max(), w2.min())


#6. poly fit to skeleton brightness curve

i = np.arange(0, len(x), 1)
order = int(length/5)
z = np.polyfit(x, B, order)
p = np.poly1d(z)

#get predicted y value (ynew) from the fitted curve for given x
ynew = p(x)
fit = np.vstack((x, ynew)).T

#get 1st order derivative of the curve
d = np.polyder(p)
dx = d(x)
#der = np.vstack((x, dx)).T
#print(dx)

#find total number of max+min, check how many times the first derivative changes sign
maxmin = 0
num_min = 0
num_max = 0
for s in range (0, len(dx)-1):
    derivative_now = dx[s]
    derivative_next = dx[s+1]
    product = derivative_now * derivative_next
    if product < 0:
        maxmin = maxmin+ 1
        if derivative_now > 0 and derivative_next < 0:
            num_max = num_max + 1
        if derivative_now < 0 and derivative_next > 0:
            num_min = num_min + 1

#print(maxmin, num_max, num_min)
feat_list.append(maxmin)
feat_list.append(num_max)
feat_list.append(num_min)


#N = (abs(dx) < 0.001).sum()
#print(N)

#get 2nd order derivative of the curve
#dd = np.polyder(p,2)
#ddx = dd(x)
#print(ddx)
#outname = str(in_arg)+'.band_fit'
#np.savetxt(outname, fit)
#outname2 = str(in_arg)+'.band_fit_der'
#np.savetxt(outname2, der)

#7. get std, max width/length ration, autocorrelation and fft of shape curve
std_shape = np.std(S)
feat_list.append(std_shape)

max_width = np.max(S)
max_w_to_l = max_width/length
feat_list.append(max_w_to_l)

autoS = auto_corr(S)
feat_list.append(np.mean(autoS))
feat_list.append(np.max(autoS))
feat_list.append(np.std(autoS))


v = np.fft.fft(S)
v2 = np.absolute(v)

for t in range (0, terms):
    feat_list.append(v2[t])

feat_list.append(np.mean(v2))
feat_list.append(np.std(v2))


feat_list.append(tag)
feat_list.append('\n')
#print(feat_list)
#print(feat_list)
#feat_array = np.asarray(feat_list)
#feat_array2 = feat_array.T
outname = str(in_arg)+'.feat'
outfile = open(outname, 'w')
outfile.write(" ".join(map(str,feat_list)))




