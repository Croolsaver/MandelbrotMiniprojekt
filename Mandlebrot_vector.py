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

iter_max = 100

tolerance = 2
p_re = 500
p_im = 500




def C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im):
    """
    

    Parameters
    ----------
    re_start : TYPE
        DESCRIPTION.
    re_stop : TYPE
        DESCRIPTION.
    im_start : TYPE
        DESCRIPTION.
    im_stop : TYPE
        DESCRIPTION.
    p_re : TYPE
        DESCRIPTION.
    p_im : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    >>> C_mesh(0, 1, 0, 1, 2, 2)
    array([[0.+1.j, 1.+1.j],
           [0.+0.j, 1.+0.j]])
    
    >>> C_mesh(0, 1, 0, 1, 3, 3)
    array([[0. +1.j , 0.5+1.j , 1. +1.j ],
           [0. +0.5j, 0.5+0.5j, 1. +0.5j],
           [0. +0.j , 0.5+0.j , 1. +0.j ]])
    
    >>> C_mesh(0, 0.5, 0, 0.5, 2, 2)
    array([[0. +0.5j, 0.5+0.5j],
           [0. +0.j , 0.5+0.j ]])
    
    >>> C_mesh(0, -1, 0, -1, 2, 2)
    array([[ 0.-1.j, -1.-1.j],
           [ 0.+0.j, -1.+0.j]])

    >>> C_mesh(0, 1, 0, 1, 2, 1)
    Traceback (most recent call last):
        ...
    ValueError: p_re and p_im must be greater than 1
    
    >>> C_mesh(0, 1, 0, 1, 0, 2)
    Traceback (most recent call last):
        ...
    ValueError: p_re and p_im must be greater than 1

    >>> C_mesh(0, 0.5j, 0, 1+0.5j, 2, 2)
    Traceback (most recent call last):
        ...
    ValueError: re_start, re_stop, im_start, im_stop cannot be complex
    """
    if p_re <= 1 or p_im <= 1:
        raise ValueError("p_re and p_im must be greater than 1")
    if np.any(np.iscomplex((re_start, re_stop, im_start, im_stop))):
        raise ValueError("re_start, re_stop, im_start, im_stop cannot be complex")
    C_re_row = np.linspace(np.real(re_start), np.real(re_stop), p_re)
    C_re = np.tile(C_re_row, (p_re, 1))
    C_im_col = np.linspace(np.real(im_stop), np.real(im_start), p_im)
    C_im = np.tile(C_im_col, (p_re, 1)).T
    return C_re+1j*C_im

def iota(c,tolerance,iter_max):
    z=0+0j
    for i in range(1,iter_max):
        z=z**2 + c
        if abs(z)>tolerance:
            return i
    return iter_max

def iota_vector(C, tolerance, iter_max):
    z = np.zeros((p_re, p_im, iter_max), dtype=complex)
    truth_table = np.zeros((p_re, p_im, iter_max), dtype=int)
    for i in range(1, iter_max):
        z[:,:,i] = z[:,:,i-1] + C
        truth_table[:,:,i] = (np.abs(z[:,:,i])>tolerance)*i
    index = np.array([[np.searchsorted(truth_table[m,n,:], 1) for m in range(p_im)] for n in range(p_re)])
    return index/iter_max

def M_map(Iter, iter_max):
    return Iter/iter_max

if __name__ == "__main__":
    import doctest
    doctest.testmod()

C=C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im)
M=np.zeros((p_re,p_im))

#M = [[iota(c, tolerance, iter_max)/iter_max for c in c_row] for c_row in C]

index = iota_vector(C, tolerance, iter_max)


#Plot mandelbrot_set
plt.pcolormesh(np.linspace(re_start,re_stop,p_re),np.linspace(im_start,im_stop,p_im),index,cmap=plt.cm.hot)