# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:08:11 2020

@authors: Daniel Van Diepen & Dennis Grøndahl Andersen
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py

# Initialization
RE_INTERVAL = [-2.0, 1.0]
IM_INTERVAL = [-1.5, 1.5]

ITER_MAX = 100

TOLERANCE = 2
P_RE = 1000
P_IM = 1000


def c_mesh(re_interval, im_interval, p_re, p_im):
    """
    Generates mesh of p evenly spaced complex points in the intervals
    [re_interval[0], re_interval[1]], [im_interval[0], im_interval[1]]

    Parameters
    ----------
    re_interval : float, float
        Start and end of real interval.
    im_interval : float, float
        Start and end of imaginary interval.
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
    c_re = np.zeros((p_re, p_im))
    c_im = np.zeros((p_re, p_im))
    re_stepsize = (re_stop-re_start)/(p_re-1)
    im_stepsize = (im_stop-im_start)/(p_im-1)
    for i in range(p_re):
        c_re[:, i] = re_start+i*re_stepsize
    for k in range(p_im):
        c_im[k, :] = im_stop-k*im_stepsize
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


def m_map(iter_stop, iter_max):
    """
    Maps iota to the interval [0,1] by dividing with iter_max.


    Parameters
    ----------
    iter_stop : integer
        The iteration iota stopped at.
    iter_max : integer
        The max amount of iterations in iota.

    Returns
    -------
    float
        Normalized M in interval [0,1].

    >>> m_map(1, 2)
    0.5

    >>> m_map(1, 0)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero

    """
    return iter_stop/iter_max


if __name__ == "__main__":
    import doctest
    doctest.testmod()

TIME_START = time.time()
C_MESH = c_mesh(RE_INTERVAL, IM_INTERVAL, P_RE, P_IM)
m_mesh_matrix = np.zeros((P_RE, P_IM))
for m in range(P_RE):
    for n in range(P_IM):
        iteration = iota(C_MESH[m, n], TOLERANCE, ITER_MAX)
        m_mesh_matrix[m, n] = m_map(iteration, ITER_MAX)

TIME_EXEC = time.time()-TIME_START
# Print execution time
print(f"Time taken: {TIME_EXEC:.4f}")

# Plot mandelbrot_set
RE_VALUES = np.linspace(*RE_INTERVAL, P_RE)
IM_VALUES = np.linspace(*IM_INTERVAL, P_IM)
plt.figure(figsize=(10,10))
plt.pcolormesh(RE_VALUES, IM_VALUES, m_mesh_matrix, cmap=cm.get_cmap("hot"))
plt.xlabel(r"$\mathfrak{R}[c]$")
plt.ylabel(r"$\mathfrak{I}[c]$")
plt.savefig("mandelbrot_naive.png", dpi=300)

with h5py.File("mandelbrot_naive.hdf5", "w") as data_file:
    data_file.create_dataset("m_mesh", data=m_mesh_matrix)
    data_file.create_dataset("re_interval", data=RE_INTERVAL)
    data_file.create_dataset("im_interval", data=IM_INTERVAL)
    data_file.create_dataset("tolerance", data=TOLERANCE)
    data_file.create_dataset("iter_max", data=ITER_MAX)
