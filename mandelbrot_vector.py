# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:08:11 2020

@authors: Daniel Van Diepen & Dennis Grøndahl Andersen
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Initialization
RE_INTERVAL = [-2.0, 1.0]
IM_INTERVAL = [-1.5, 1.5]

ITER_MAX = 100

TOLERANCE = 2
P_RE = 1000
P_IM = 1000



def c_mesh(re_interval, im_interval, p_re, p_im):
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

    >>> c_mesh([0, 1], [0, 1], 2, 2)
    array([[0.+1.j, 1.+1.j],
           [0.+0.j, 1.+0.j]])

    >>> c_mesh([0, 1], [0, 1], 3, 3)
    array([[0. +1.j , 0.5+1.j , 1. +1.j ],
           [0. +0.5j, 0.5+0.5j, 1. +0.5j],
           [0. +0.j , 0.5+0.j , 1. +0.j ]])

    >>> c_mesh([0, 0.5], [0, 0.5], 2, 2)
    array([[0. +0.5j, 0.5+0.5j],
           [0. +0.j , 0.5+0.j ]])

    >>> c_mesh([0, -1], [0, -1], 2, 2)
    array([[ 0.-1.j, -1.-1.j],
           [ 0.+0.j, -1.+0.j]])

    >>> c_mesh([0, 1], [0, 1], 2, 1)
    Traceback (most recent call last):
        ...
    ValueError: p_re and p_im must be greater than 1

    >>> c_mesh([0, 1], [0, 1], 0, 2)
    Traceback (most recent call last):
        ...
    ValueError: p_re and p_im must be greater than 1

    >>> c_mesh([0, 0.5j], [0, 1+0.5j], 2, 2)
    Traceback (most recent call last):
        ...
    ValueError: re_start, re_stop, im_start, im_stop cannot be complex
    """
    re_start, re_stop = re_interval
    im_start, im_stop = im_interval
    if p_re <= 1 or p_im <= 1:
        raise ValueError("p_re and p_im must be greater than 1")
    if np.any(np.iscomplex((re_start, re_stop, im_start, im_stop))):
        raise ValueError("re_start, re_stop, im_start, im_stop cannot be complex")
    c_re_row = np.linspace(np.real(re_start), np.real(re_stop), p_re)
    c_re = np.tile(c_re_row, (p_re, 1))
    c_im_col = np.linspace(np.real(im_stop), np.real(im_start), p_im)
    c_im = np.tile(c_im_col, (p_re, 1)).T
    return c_re+1j*c_im

def iota(complex_point, tolerance, iter_max):
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
    z_current = 0+0j
    for i in range(1, iter_max):
        z_current = z_current**2 + complex_point
        if abs(z_current) > tolerance:
            return i
    return iter_max

def iota_vector(c_mesh, tolerance, iter_max, p_re, p_im):
    """
    

    Parameters
    ----------
    C : TYPE
        DESCRIPTION.
    tolerance : TYPE
        DESCRIPTION.
    iter_max : TYPE
        DESCRIPTION.
    p_re : TYPE
        DESCRIPTION.
    p_im : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.


    >>> [iota_vector(0, 2, n) for n in range(0, 100, 10)]
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    >>> [iota_vector(c, 2, 100) for c in np.linspace(0, 1, 11)]
    [100, 100, 100, 12, 7, 5, 4, 3, 3, 3, 3]

    >>> iota_vector(1+1j, 2, 100)
    2

    >>> [iota_vector(n+1, n, 100) for n in range(1,6)]
    [1, 1, 1, 1, 1]

    >>> iota_vector(1+1j, 1+2j, 100)
    Traceback (most recent call last):
        ...
    TypeError: '>' not supported between instances of 'float' and 'complex'

    >>> iota_vector(1+1j, 2, 100.5)
    Traceback (most recent call last):
        ...
    TypeError: 'float' object cannot be interpreted as an integer
    """
    z_current = np.zeros((p_re, p_im, iter_max), dtype=complex)
    truth_table = np.zeros((p_re, p_im, iter_max), dtype=int)
    truth_table[:,:,0] = iter_max
    for i in range(1, iter_max):
        z_current[:,:,i] = z_current[:,:,i-1]**2 + c_mesh
        tt_local  = np.abs(z_current[:,:,i]) > tolerance
        truth_table[:,:,i] = tt_local*i + np.logical_not(tt_local)*iter_max
    index = np.min(truth_table, axis=2)
    return index/iter_max

def m_map(iter_stop, iter_max):
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

    >>> m_map(1, 2)
    0.5

    >>> m_map(1, 0)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero

    """
    return iter_stop/iter_max

#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()

TIME_START = time.time()
C_MESH = c_mesh(RE_INTERVAL, IM_INTERVAL, P_RE, P_IM)
M_MESH = iota_vector(C_MESH, TOLERANCE, ITER_MAX, P_RE, P_IM)

TIME_EXEC = time.time()-TIME_START
#Print execution time
print(f"Time taken: {TIME_EXEC:.4f}")

#Plot mandelbrot_set
RE_VALUES = np.linspace(*RE_INTERVAL, P_RE)
IM_VALUES = np.linspace(*IM_INTERVAL, P_IM)
plt.pcolormesh(RE_VALUES, IM_VALUES, M_MESH, cmap=cm.get_cmap("hot"))
