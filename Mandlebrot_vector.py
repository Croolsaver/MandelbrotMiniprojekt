# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:08:11 2020

@authors: Daniel Van Diepen & Dennis Grøndahl Andersen
"""
import numpy as np
import matplotlib.pyplot as plt
#Initialization
re_start = -2.0
re_stop = 1.0
im_start = -1.5
im_stop = 1.5

Iter_max = 100

Tolerance = 2
p_re = 500
p_im = 500




def C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im):
    C_re_row = np.linspace(re_start, re_stop, p_re)
    C_re = np.tile(C_re_row, (p_re, 1))
    C_im_col = np.linspace(im_start, im_stop, p_im)
    C_im = np.tile(C_im_col, (p_re, 1)).T
    return C_re+1j*C_im

def iota(c,Tolerance,Iter_max):
    z=0+0j
    for i in range(1,Iter_max):
        z=z**2 + c
        if abs(z)>Tolerance:
            return i
    return Iter_max

def iota_vector(C, Tolerance, Iter_max):
    z = np.zeros((p_re, p_im, Iter_max), dtype=complex)
    truth_table = np.zeros((p_re, p_im, Iter_max), dtype=int)
    for i in range(1, Iter_max):
        z[:,:,i] = z[:,:,i-1] + C
        truth_table[:,:,i] = (np.abs(z[:,:,i])>Tolerance)*i
    index = np.array([[np.searchsorted(truth_table[m,n,:], 1) for m in range(p_im)] for n in range(p_re)])
    return index/Iter_max

def M_map(Iter, Iter_max):
    return Iter/Iter_max


C=C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im)
M=np.zeros((p_re,p_im))

#M = [[iota(c, Tolerance, Iter_max)/Iter_max for c in c_row] for c_row in C]

index = iota_vector(C, Tolerance, Iter_max)


#Plot mandelbrot_set
plt.pcolormesh(np.linspace(re_start,re_stop,p_re),np.linspace(im_start,im_stop,p_im),index,cmap=plt.cm.hot)