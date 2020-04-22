# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:08:11 2020

@authors: Daniel Van Diepen & Dennis Grøndahl Andersen
"""
import time
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt


# Initialization
CPU_COUNT = mp.cpu_count()

RE_INTERVAL = [-2.0, 1.0]
IM_INTERVAL = [-1.5, 1.5]

ITER_MAX = 100

TOLERANCE = 2
P_RE = 1000
P_IM = 1000

N_AVERAGE = 10


def c_mesh(re_interval, im_interval, p_re, p_im):
    """
    Generates mesh of p evenly spaced complex points in the intervals
    [re_interval[0], re_interval[1]], [im_interval[0], im_interval[1]]

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
        raise ValueError("re_start, re_stop, im_start, im_stop cannot be "
                         "complex")
    if np.any(np.iscomplex((re_start, re_stop, im_start, im_stop))):
        raise ValueError("re_start, re_stop, im_start, im_stop cannot be "
                         "complex")
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
    c : complex
        The complex number for which we wish to calculate iota.
    tolerance : float
        Threshold z should stay below.
    iter_max : integer
        Maximum amount of iterations.

    Returns
    -------
    integer
        Number of iterations for z to surpass the threshold.


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


def iota_vec(tolerance, iter_max, c_vec):
    """
    Calculates iota for a vector, and applies M-mapping by dividing by iter_max

    Parameters
    ----------
    tolerance : float
        Threshold z should stay below.
    iter_max : integer
        Maximum amount of iterations.
    c_vec : list of complex
        list of complex numbers to calculate mandelbrot values for.

    Returns
    -------
    list of float
        1d-numpy array of normalized M.

    >>> iota_vec(2, 100, range(10, 110, 10))
    array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    >>> iota_vec(2, 100, np.linspace(0, 1, 11))
    array([1.  , 1.  , 1.  , 0.12, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03])

    """
    return np.array([iota(c, tolerance, iter_max) for c in c_vec])/iter_max


def calculate_m_np(n_processors, tolerance, iter_max, complex_mesh):
    """
    Calculates M using multiprocessing with n_processors processors.

    Parameters
    ----------
    n_processors : integer
        Number of processors to use in the multiprocessing implementation.

    Returns
    -------
    2d-numpy array of float
        Mandelbrot set values corresponding to the points in complex_mesh.

    """
    pool = mp.Pool(processes=n_processors)
    iota_partial = partial(iota_vec, tolerance, iter_max)
    result = pool.map_async(iota_partial, complex_mesh)
    pool.close()
    pool.join()
    return result.get()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    times = np.zeros(CPU_COUNT)
    C_MESH = c_mesh(RE_INTERVAL, IM_INTERVAL, P_RE, P_IM)
    for j in range(CPU_COUNT):
        t_start = np.zeros(N_AVERAGE)
        t_elapsed = np.zeros(N_AVERAGE)
        for n in range(N_AVERAGE):
            t_start[n] = time.time()
            calculate_m_np(j+1, TOLERANCE, ITER_MAX, C_MESH)
            t_elapsed[n] = time.time() - t_start[n]
        times[j] = np.mean(t_elapsed)
        print("Time is for {} core(s) is {}".format(j+1, times[j]))
    plt.bar(range(1, CPU_COUNT+1), times)
    plt.title("Mandelbrot set multiprocessing")
    plt.xlabel("# processors")
    plt.ylabel("time (s)")
    plt.savefig("output/mandelbrot_multiprocessing_n_processors.pdf")
