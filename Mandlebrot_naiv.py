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
p_re = 5000
p_im = 5000




def C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im):
    C_re=np.zeros((p_re,p_im))
    C_im=np.zeros((p_re,p_im))
    re_stepsize = (re_stop-re_start)/p_re 
    im_stepsize = (im_stop-im_start)/p_im
    for i in range(p_re):
        C_re[:,i] = re_start+i*re_stepsize
    for k in range(p_im):
        C_im[k,:] = im_stop-k*im_stepsize
    return C_re+1j*C_im

def iota(c,Tolerance,Iter_max):
    z=0+0j
    for i in range(1,Iter_max):
        z=z**2 + c
        if abs(z)>Tolerance:
            return i
    return Iter_max
        
def M_map(Iter, Iter_max):
    return Iter/Iter_max


C=C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im)
M=np.zeros((p_re,p_im))
for m in range(p_re):
    for n in range(p_im):
        Iter = iota(C[m,n],Tolerance,Iter_max)
        M[m,n]=M_map(Iter,Iter_max)
        
       
#Plot mandelbrot_set
plt.pcolormesh(np.linspace(re_start,re_stop,p_re),np.linspace(im_start,im_stop,p_im),M,cmap=plt.cm.hot)