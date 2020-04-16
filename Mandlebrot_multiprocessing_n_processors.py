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

iter_max = 100

tolerance = 2
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

def iota(tolerance,iter_max, c):
    z=0+0j
    for i in range(1,iter_max):
        z=z**2 + c
        if abs(z)>tolerance:
            return i
    return iter_max

def iota_vec(tolerance, iter_max, c_vec):
    return np.array([iota(tolerance, iter_max, c) for c in c_vec])
        
def M_map(Iter, iter_max):
    return Iter/iter_max

def calculate_M_mp(n_processors):
    pool = mp.Pool(processes=n_processors)
    iota_partial = partial(iota_vec, tolerance, iter_max)
    result = pool.map_async(iota_partial, C)
    pool.close()
    pool.join()
    return result.get()

C=C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im)

if __name__ == '__main__':
    times = np.zeros(cpu_count)
    for i in range(cpu_count):
        t_start = time.time()
        calculate_M_mp(i+1)
        times[i] = time.time() - t_start
<<<<<<< HEAD

=======
        print ("Time is for {} core(s) is {}".format(i+1,times[i]))
>>>>>>> 9274ef9d1f72c183335f63f22eacd6ed55a1074c
    plt.plot(range(1,cpu_count+1), times)
