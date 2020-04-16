# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:08:11 2020

@authors: Daniel Van Diepen & Dennis Grøndahl Andersen
"""
import numpy as np
import matplotlib.pyplot as plt
import time
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
    """
    Generates mesh of p evenly spaced complex points in the intervals [re_start].....

    Parameters
    ----------
    re_start : float
        Start of real interval.
    re_stop : float
        End of real interval.
    im_start : float
        Start of imaginary interval.
    im_stop : float
        End of imaginary interval.
    p_re : integer > 1
        Number of points in the real interval.
    p_im : integer > 1
        Number of points in the imaginary interval.

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
    C_re=np.zeros((p_re,p_im))
    C_im=np.zeros((p_re,p_im))
    re_stepsize = (re_stop-re_start)/(p_re-1) 
    im_stepsize = (im_stop-im_start)/(p_im-1)
    for i in range(p_re):
        C_re[:,i] = re_start+i*re_stepsize
    for k in range(p_im):
        C_im[k,:] = im_stop-k*im_stepsize
    return C_re+1j*C_im

def iota(c,tolerance,iter_max):
    """
    Calculates Mandelbrot iota function

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    tolerance : TYPE
        DESCRIPTION.
    iter_max : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.


    >>> [iota(0, 2, n) for n in range(0, 100, 10)]
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    >>> [iota(c, 2, 100) for c in np.linspace(0, 1, 11)]
    [100, 100, 100, 12, 7, 5, 4, 3, 3, 3, 3]
    
    >>> iota(1+1j, 2, 100)
    2
    
    >>> [iota(n+1, n, 100) for n in range(1,6)]
    [1, 1, 1, 1, 1]
    
    >>> iota(1+1j, 1+2j, 100)
    Traceback (most recent call last):
        ...
    TypeError: '>' not supported between instances of 'float' and 'complex'
    
    >>> iota(1+1j, 2, 100.5)
    Traceback (most recent call last):
        ...
    TypeError: 'float' object cannot be interpreted as an integer
    """
    z=0+0j
    for i in range(1,iter_max):
        z=z**2 + c
        if abs(z)>tolerance:
            return i
    return iter_max
        
def M_map(Iter, iter_max):
    """
    

    Parameters
    ----------
    Iter : TYPE
        DESCRIPTION.
    iter_max : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    >>> M_map(1, 2)
    0.5
    
    >>> M_map(1, 0)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero

    """
    return Iter/iter_max

if __name__ == "__main__":
    import doctest
    doctest.testmod()

time_start=time.time()
C=C_mesh(re_start,re_stop,im_start,im_stop,p_re,p_im)
M=np.zeros((p_re,p_im))
for m in range(p_re):
    for n in range(p_im):
        Iter = iota(C[m,n],tolerance,iter_max)
        M[m,n]=M_map(Iter,iter_max)
        
time_exec=time.time()-time_start       

#Print execution time
print (f"time taken: {time_exec:.4f}")
#Plot mandelbrot_set
plt.pcolormesh(np.linspace(re_start,re_stop,p_re),np.linspace(im_start,im_stop,p_im),M,cmap=plt.cm.hot)