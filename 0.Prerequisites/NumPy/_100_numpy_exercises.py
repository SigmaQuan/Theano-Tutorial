# search keyword "numpy-100" from github
# https://github.com/rougier/numpy-100
# deeplearning.net/software/tutorial/numpy.html
# 1. Import the numpy package under the name np (*)
import numpy as np


# 2. Print the numpy version and the configuration (*)
print(np.__version__)
# np.show_config()
# 1.11.0
# lapack_opt_info:
#     libraries = ['openblas', 'openblas']
#     library_dirs = ['/usr/lib']
#     define_macros = [('HAVE_CBLAS', None)]
#     language = c
# blas_opt_info:
#     libraries = ['openblas', 'openblas']
#     library_dirs = ['/usr/lib']
#     define_macros = [('HAVE_CBLAS', None)]
#     language = c
# openblas_info:
#     libraries = ['openblas', 'openblas']
#     library_dirs = ['/usr/lib']
#     define_macros = [('HAVE_CBLAS', None)]
#     language = c
# openblas_lapack_info:
#     libraries = ['openblas', 'openblas']
#     library_dirs = ['/usr/lib']
#     define_macros = [('HAVE_CBLAS', None)]
#     language = c
# blas_mkl_info:
#   NOT AVAILABLE

# 3. Create a null vector of size 10 (*)
z = np.zeros(10)
print z
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

# 4. How to get the documentation of the numpy add fucntion from the command line? (*)
# python -c "import numpy; numpy.info(numpy.add)"

# add(x1, x2[, out])
#
# Add arguments element-wise.
#
# Parameters
# ----------
# x1, x2 : array_like
#     The arrays to be added.  If ``x1.shape != x2.shape``, they must be
#     broadcastable to a common shape (which may be the shape of one or
#     the other).
#
# Returns
# -------
# add : ndarray or scalar
#     The sum of `x1` and `x2`, element-wise.  Returns a scalar if
#     both  `x1` and `x2` are scalars.
#
# Notes
# -----
# Equivalent to `x1` + `x2` in terms of array broadcasting.
#
# Examples
# --------
# >>> np.add(1.0, 4.0)
# 5.0
# >>> x1 = np.arange(9.0).reshape((3, 3))
# >>> x2 = np.arange(3.0)
# >>> np.add(x1, x2)
# array([[  0.,   2.,   4.],
#        [  3.,   5.,   7.],
#        [  6.,   8.,  10.]])

# 5. Create a null vector of size 10 but the fifth value which is 1 (*)
z = np.zeros(10)
z[5] = 1
print z
# [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]

# 6. Create a vector with values ranging from 10 to 49 (*)
z = np.arange(10, 50)
print z
# [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
#  35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]

# 7. Reverse a vector (first element becomes last) (*)
z = np.arange(50)
z = z[::-1]
print z
# [49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25
#  24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0]

# 8. Create a 3x3 matrix with values ranging from 0 to 8 (*)
# z = np.arange(9).reshape(3, 3)
# shape = (3, 3)
# z = np.arange(9).reshape(shape)
# print z
# # [[0 1 2]
# #  [3 4 5]
# #  [6 7 8]]
z = np.arange(9)[::-1].reshape(3, 3)
print z
# [[8 7 6]
#  [5 4 3]
#  [2 1 0]]

# 9. Find indices of non-zero elements from [1, 2, 0, 0, 4, 0] (*)
non_zeros = np.nonzero([1, 2, 0, 0, 4, 0])
print non_zeros
# (array([0, 1, 4]),)

# 10. Create a 3x3 identity matrix (*)
z = np.eye(3)
print z
# [[ 1.  0.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]]

# 11. Create a 3x3x3 array with random values (*)
shape = (3, 3, 3)
z = np.random.random(shape)
print z
# [[[  4.86460562e-01   6.45039650e-01   9.60110784e-01]
#   [  6.12233386e-01   3.66104402e-01   6.17026750e-01]
#   [  3.79693250e-01   3.77667544e-01   6.12043488e-01]]
#
#  [[  8.47145800e-01   3.69164508e-05   6.28662837e-02]
#   [  4.54753054e-01   3.76770710e-01   8.72098580e-02]
#   [  2.62050244e-01   5.47694713e-01   2.51709202e-02]]
#
#  [[  5.17994546e-01   5.50959149e-01   8.69922205e-01]
#   [  2.49656698e-01   9.32779606e-01   8.89092411e-01]
#   [  2.64493932e-01   1.78998771e-01   7.37046902e-01]]]

# 12. Create a 10x10 array with random values and find the minmum and maximum values (*)
shape = (10, 10)
z = np.random.random(shape)
# print np.max(np.max(z)), np.min(np.min(z))
# print np.max(z), np.min(z)
# 0.985055771164 0.0201160702491
print z.max(), z.min()
# 0.996388368542 0.00253072245619

# 13. Create a random vector of size 30 and find the mean value (*)
z = np.random.random(30)
print z
print z.mean()
# [ 0.65578714  0.9302428   0.89284316  0.68967148  0.77417002  0.56841976
#   0.43681135  0.93721967  0.79525303  0.2685454   0.28189049  0.20862684
#   0.44817271  0.89738212  0.04721653  0.66438186  0.49760447  0.24259219
#   0.69249514  0.95550059  0.07747862  0.41225455  0.04789703  0.25900736
#   0.58216405  0.05766379  0.5453765   0.11793314  0.38438833  0.01451072]
# 0.479450027645

# 14. Create a 2d array with 1 on the border and 0 inside (*)
shape = (5, 5)
z = np.ones(shape)
print z
z[1:-1, 1:-1] = 0
print z
# [[ 1.  1.  1.  1.  1.]
#  [ 1.  1.  1.  1.  1.]
#  [ 1.  1.  1.  1.  1.]
#  [ 1.  1.  1.  1.  1.]
#  [ 1.  1.  1.  1.  1.]]
# [[ 1.  1.  1.  1.  1.]
#  [ 1.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  1.]
#  [ 1.  1.  1.  1.  1.]]

# 15. What is the result of the following expression? (*)
a = 0 * np.nan
b = (np.nan == np.nan)
c = (np.inf > np.nan)
d = (np.nan - np.nan)
e = (0.3 == 3 * 0.1)
print a, b, c, d, e
# nan False False nan False


# 16. Create a 5x5 matrix with values 1, 2, 3, 4 just below the diagonal (*)
z = np.diag(1+np.arange(4), k=-1)
print z
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]

# 17. Create a 8x8 matrix and fill it with a checkerboard pattern (*)
z = np.zeros((8, 8), dtype=int)
z[1::2, ::2] = 1
z[::2, 1::2] = 1
print z
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]

# 18.