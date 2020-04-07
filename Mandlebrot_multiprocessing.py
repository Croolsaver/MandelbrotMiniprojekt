# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:08:11 2020

@authors: Daniel Van Diepen & Dennis Grøndahl Andersen
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os, time
from functools import partial

cpu_count = mp.cpu_count()
#Initialization
re_start = -2.0
re_stop = 1.0
im_start = -1.5
im_stop = 1.5

Iter_max = 100

Tolerance = 2
p_re = 1000
p_im = 1000

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

def _f(d):
    # Defines the f(d) function (f(d) is a single task)
    time.sleep(float(d))
    pid = os.getpid()
    print(" _f argument: {:2d}, process id: {:7d} ".format(d, pid))
    return pid

def iota(Tolerance,Iter_max, c):
    z=0+0j
    for i in range(1,Iter_max):
        z=z**2 + c
        if abs(z)>Tolerance:
            return i
    return Iter_max

def iota_vec(Tolerance, Iter_max, c_vec):
    return np.array([iota(Tolerance, Iter_max, c) for c in c_vec])
        
def M_map(Iter, Iter_max):
    return Iter/Iter_max

C=C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im)



if __name__ == '__main__':
    pool = mp.Pool(processes=cpu_count)
    #iota_partial = partial(iota, Tolerance, Iter_max)
    #result = pool.map_async(iota_partial, C.flatten())
    iota_partial = partial(iota_vec, Tolerance, Iter_max)
    result = pool.map_async(iota_partial, C)
    pool.close()
    pool.join()
    #M = np.reshape(result.get(), (p_re, p_im))
    M = result.get()

profile.print_stats()
        
       
#Plot mandelbrot_set
plt.pcolormesh(np.linspace(re_start,re_stop,p_re),np.linspace(im_start,im_stop,p_im),M,cmap=plt.cm.hot)