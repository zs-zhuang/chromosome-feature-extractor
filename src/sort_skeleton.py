#! /usr/bin/python3

import os, sys, math, string
import pandas as pd
import numpy as np


#########################################################################################

#read in text file containing unsorted skeleton coordinates as numpy array
in_arg = sys.argv[1]
in_name = str(in_arg) + '.txt1'
coor = np.loadtxt(in_name, dtype=np.int)

l, c = coor.shape

#convert numpy array to set for lookup purpose
t = map(tuple, coor)
s = set(t)

#print(s)

#find the two terminal pixels of the skeleton, which would only have one connecting neighboring pixel instead of two
#save to a list called term_list
term_list = list()

for i in range(0, l):
    count = 0
    m = coor[i][0]
    n = coor[i][1]
    pos = (m, n)

    nb1 = (m-1,n)
    nb2 = (m+1, n)
    nb3 = (m, n-1)
    nb4 = (m, n+1)
    nb5 = (m-1,n-1)
    nb6 = (m-1,n+1)
    nb7 = (m+1, n-1)
    nb8 = (m+1, n+1)

    if nb1 in s:
        count = count + 1
    if nb2 in s:
        count = count + 1
    if nb3 in s:
        count = count + 1
    if nb4 in s:
        count = count + 1
    if nb5 in s:
        count = count + 1
    if nb6 in s:
        count = count + 1
    if nb7 in s:
        count = count + 1
    if nb8 in s:
        count = count + 1
    
    #print(count)
    if count == 1:
        term_list.append(pos)
    if count > 2:
        print(pos, str(in_arg)+": skeleton is deformed, has branch point")
        sys.exit(1)

#print(term_list)
if len(term_list) != 2:
    print(str(in_arg)+": skeleton is deformed, has more than two terminals")
    sys.exit(1)


#find out which terminal is the top(smaller row value), which one is the bottom (bigger row value)
#initialize a numpy array the length of the skeleton and put skeleton pixels into it with the correct order
#starting with the two terminals

used = set([])
sort_array = np.zeros((l,2))
current = np.zeros((1,2))

x1 = term_list[0][0]
x2 = term_list[1][0]

# Put terminals into the sort_array
if x1 < x2:
    sort_array[0] = term_list[0]
    sort_array[l-1] = term_list[1]
    current = term_list[0]

if x1 >= x2:
    sort_array[0] = term_list[1]
    sort_array[l-1] = term_list[0]
    current = term_list[1]


#print(sort_array)
#print(current)
used.add(term_list[0])
used.add(term_list[1])
#print(used)

# Put non-terminal pixels in the correct order into the sort_array

def get_my_neighbor(x, y):
    nb1 = (x-1,y)
    nb2 = (x+1, y)
    nb3 = (x, y-1)
    nb4 = (x, y+1)
    nb5 = (x-1,y-1)
    nb6 = (x-1,y+1)
    nb7 = (x+1, y-1)
    nb8 = (x+1, y+1)

    if nb1 not in used and nb1 in s:
        sort_array[c] = nb1
        used.add(nb1)
        current = nb1
    if nb2 not in used and nb2 in s:
        sort_array[c] = nb2
        used.add(nb2)
        current = nb2
    if nb3 not in used and nb3 in s:
        sort_array[c] = nb3
        used.add(nb3)
        current = nb3
    if nb4 not in used and nb4 in s:
        sort_array[c] = nb4
        used.add(nb4)
        current = nb4
    if nb5 not in used and nb5 in s:
        sort_array[c] = nb5
        used.add(nb5)
        current = nb5
    if nb6 not in used and nb6 in s:
        sort_array[c] = nb6
        used.add(nb6)
        current = nb6
    if nb7 not in used and nb7 in s:
        sort_array[c] = nb7
        used.add(nb7)
        current = nb7
    if nb8 not in used and nb8 in s:
        sort_array[c] = nb8
        used.add(nb8)
        current = nb8
    return current

c = 1
for c in range (1, l-1):
    current = get_my_neighbor(current[0], current[1])
    #print(used)
    #print(c)
    assert(len(used) == c + 2)

#print(sort_array)
#print(current)
#print(used)
#print(c)



np.savetxt(str(in_arg)+".txt2", sort_array, fmt='%1.0i')
