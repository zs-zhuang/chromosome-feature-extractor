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
name = 'skeleton_'+str(in_arg)+'.png'
a = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
(thresh, b) = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
######################################################################################

#perform median filter (get rid of specks or salt pepper noise)
#m = scipy.ndimage.filters.median_filter(b,size=2,footprint=None,output=None,mode='reflect',cval=0.0,origin=0)


height, width = b.shape
#print(width, height)


d_white = {}
my_list = []

index = 1

# Go through every pixel and write out location of each white pixel and assign an unique index to each location
for m in range (0, height):
    for n in range (0, width):
        if b[m][n] == 255:
            location = (m, n)
            #location = str(m)+' '+str(n)
            my_list.append(location)
            d_white[(location)] = index
            index = index + 1

#print(len(d_white))
#print(len(my_list))

# Go through all the white pixels, if they are connected, their index value will be updated to the same index

def check_neighbors(neighbor, delta, index_now, location_now):
    if neighbor in d_white:
        if index_now < d_white[neighbor]:
            d_white[neighbor] = index_now
            delta = delta + 1
        if index_now > d_white[neighbor]:
            d_white[location_now] =  d_white[neighbor]
            delta = delta + 1
    return delta


delta = -1
while delta != 0:
    delta = 0
    for e in range (0, len(my_list)):
        location_now = my_list[e]
        x = location_now[0]
        y = location_now[1]
        #x = int(location_now.split()[0])
        #y = int(location_now.split()[1])
        index_now = d_white[(location_now)]

        neighbor1 = (x-1, y-1)
        neighbor2 = (x-1, y)
        neighbor3 = (x-1, y+1)
        neighbor4 = (x, y-1)
        neighbor5 = (x, y+1)
        neighbor6 = (x+1, y-1)
        neighbor7 = (x+1, y)
        neighbor8 = (x+1, y+1)

        delta = check_neighbors(neighbor1, delta, index_now, location_now)
        delta = check_neighbors(neighbor2, delta, index_now, location_now)
        delta = check_neighbors(neighbor3, delta, index_now, location_now)
        delta = check_neighbors(neighbor4, delta, index_now, location_now)
        delta = check_neighbors(neighbor5, delta, index_now, location_now)
        delta = check_neighbors(neighbor6, delta, index_now, location_now)
        delta = check_neighbors(neighbor7, delta, index_now, location_now)
        delta = check_neighbors(neighbor8, delta, index_now, location_now)

        #print(delta)


# After final iteration
# convert dictionary white to a list, white_list
# loop through white_list and count how many pixels belong to a given index
# write white_list out as a two column data file: location vs. index


white_list = []
for key, value in d_white.items():
    temp = (key, value)
    white_list.append(temp)

#out_data = open('white_pix_location_vs_colony_index', 'w')
#for w in range (0, len(white_list)):
#    row = white_list[w]
#out_data.write(str(row)+'\n')


d_index = set([])
for p in range (0, len(white_list)):
    row = white_list[p]
    index = row[1]
    if index not in d_index:
        d_index.add(index)

#out_report = open('report_'+str(in_arg), 'w')
print('total number of chromosome detected: '+str(len(d_index)))

if len(d_index) != 1:
    print("More than one chromsome detected when there should only be one")
    sys.exit(1)

#out_report.write('total number of chromosome detected: '+str(len(d_index))+'\n')

for value in d_index:
    my_key = []
    index = value
    for q in range (0, len(white_list)):
        row2 = white_list[q]
        xy = row2[0]
        current_index = row2[1]
        if current_index == index:
            my_key.append(xy)
    size = len(my_key)
    #key_sort = sorted(my_key, key=natural_key)
    key_sort = sorted(my_key)
    #print("chromosome_length_is "+str(size))
    #out_report.write("chromosome_length_is "+str(size)+'\n')
    #chromosome_name = 'C'+str(index)+'_size_'+str(size)+'.txt1'
    #chromosome_name = 'size_'+str(size)+'_C'+str(index)+'.txt1'
    chromosome_name = str(in_arg)+'.txt1'
    out_file = open(chromosome_name, 'w')
    for s in range (0, len(my_key)):
        entry = key_sort[s]
        out_file.write(str(entry[0])+' '+str(entry[1])+'\n')
        #out_file.write(str(entry)+'\n')

