# MandelbrotMiniprojekt
Miniproject for numerical scientific computing Mattek6-4

## Included Scripts
```
mandelbrot_naive.py
mandelbrot_vector.py
mandelbrot_multiprocessing.py
mandelbrot_multiprocessing_n_processors.py
```

- The naive implementation is found in `mandelbrot_naive.py`.
- The vectorized implementation is found in `mandelbrot_vector.py`
- The multiprocessing implementation is found in `mandelbrot_multiprocessing.py`
- A script to illustrate the time taken with n processors is found in `mandelbrot_multiprocessing_n_processors.py`

## Accompanying Material
For each of the scripts, the output files are included with the same name as the corresponding script. E.g. for `mandelbrot_naive.py`, see `mandelbrot_naive.png` and `mandelbrot_naive.hdf5`

## Prerequesites
The following Python packages are needed to run the scripts
```
time
multiprocessing
functools
numpy
matplotlib
h5py
```

## Usage
Modify constants at the top of the script under `Initialization` to your liking 
```python
RE_INTERVAL = [-2.0, 1.0]
IM_INTERVAL = [-1.5, 1.5]

ITER_MAX = 100

TOLERANCE = 2
P_RE = 1000
P_IM = 1000
```
In the multiprocessing implementation, you can additionally change the amount of processors to your liking
```python
CPU_COUNT = mp.cpu_count()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
