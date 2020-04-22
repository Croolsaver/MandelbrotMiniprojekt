# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:08:11 2020.

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
    Generate mesh of p evenly spaced complex points in the intervals
    [re_interval[0], re_interval[1]], [im_interval[0], im_interval[1]].

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
    complex 2d-array
        Mesh of evenly spaced complex points in the specified intervals.


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
    c_re_row = np.linspace(np.real(re_start), np.real(re_stop), p_re)
    c_re = np.tile(c_re_row, (p_re, 1))
    c_im_col = np.linspace(np.real(im_stop), np.real(im_start), p_im)
    c_im = np.tile(c_im_col, (p_re, 1)).T
    return c_re+1j*c_im


def iota_vector(complex_mesh, tolerance, iter_max, p_re, p_im):
    """
    Calculates the complex mapping which generates the mandelbrot set.

    Parameters
    ----------
    complex_mesh : complex 2d-array
        c-mesh of evenly spaced complex numbers.
    tolerance : float
        Threshold z should stay below.
    iter_max : integer
        Maximum amount of iterations.
    p_re : integer
        Number of values in the real dimension.
    p_im : integer
        Number of values in the imaginary dimension.

    Returns
    -------
    2d-array of integers
        The matrix M of values in the mandelbrot set, normalized for the max
        number of iterations.


    >>> [iota_vector(0, 2, n, 1, 1) for n in range(20, 120, 20)]
    [array([[1.]]), array([[1.]]), array([[1.]]), array([[1.]]), array([[1.]])]

    >>> iota_vector(np.linspace(0, 1, 11), 2, 100, 1, 11)
    array([[1.  , 1.  , 1.  , 0.12, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03]])

    >>> iota_vector(np.array([1+1j, 0+2j]), 2, 100, 1, 2)
    array([[0.02, 0.02]])

    >>> [iota_vector(n+1, n, 100, 1, 1) for n in range(1,6)]
    [array([[0.01]]), array([[0.01]]), array([[0.01]]), array([[0.01]]), \
array([[0.01]])]

    >>> iota_vector(np.array([[1,2],[3,4]]), 2, 100, 2, 2)
    array([[0.03, 0.02],
           [0.01, 0.01]])

    >>> iota_vector(1+1j, 2, 100.5, 1, 1)
    Traceback (most recent call last):
        ...
    TypeError: 'float' object cannot be interpreted as an integer
    """
    z_current = np.zeros((p_re, p_im, iter_max), dtype=complex)
    truth_table = np.zeros((p_re, p_im, iter_max), dtype=int)
    truth_table[:, :, 0] = iter_max
    for i in range(1, iter_max):
        z_current[:, :, i] = z_current[:, :, i-1]**2 + complex_mesh
        tt_local = np.abs(z_current[:, :, i]) > tolerance
        truth_table[:, :, i] = tt_local*i + np.logical_not(tt_local)*iter_max
    index = np.min(truth_table, axis=2)
    return index/iter_max


if __name__ == "__main__":
    import doctest
    doctest.testmod()

TIME_START = time.time()
C_MESH = c_mesh(RE_INTERVAL, IM_INTERVAL, P_RE, P_IM)
M_MESH = iota_vector(C_MESH, TOLERANCE, ITER_MAX, P_RE, P_IM)

TIME_EXEC = time.time()-TIME_START
# Print execution time
print(f"Time taken: {TIME_EXEC:.4f}")

# Plot mandelbrot_set
RE_VALUES = np.linspace(*RE_INTERVAL, P_RE)
IM_VALUES = np.linspace(*IM_INTERVAL, P_IM)
plt.figure(figsize=(10, 10))
plt.pcolormesh(RE_VALUES, IM_VALUES, M_MESH, cmap=cm.get_cmap("hot"))
plt.xlabel(r"$\mathfrak{R}[c]$")
plt.ylabel(r"$\mathfrak{I}[c]$")
plt.savefig("output/mandelbrot_vector.png", dpi=300)

with h5py.File("output/mandelbrot_vector.hdf5", "w") as data_file:
    data_file.create_dataset("m_mesh", data=M_MESH)
    data_file.create_dataset("re_interval", data=RE_INTERVAL)
    data_file.create_dataset("im_interval", data=IM_INTERVAL)
    data_file.create_dataset("tolerance", data=TOLERANCE)
    data_file.create_dataset("iter_max", data=ITER_MAX)
