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


# 16. Create a 5x5 matrix with values 1, 2, 3, 4 just below the
# diagonal (*)
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

# 18. Consider a (6, 7, 8) shape array, what is the index (x, y, z) of the
# 100-th element?
print(np.unravel_index(100, (6, 7, 8)))
# print 1 * (7 * 8) + 5 * 8 + 4
# a = np.unravel_index(100, (6, 7, 8))
# print a
# print a[0] * (7 * 8) + a[1] * 8 + a[2]

# 19. Create a checkerboard 8x8 matrix using the tile function (*)
z = np.tile(np.array([[20, 10], [10, 20]]), (4, 4))
print z


# 20. Normalize a 5x5 random matrix (*)
z = np.random.random((5, 5)) * 10
print z
z_max, z_min = z.max(), z.min()
print z_max, z_min
range = z_max - z_min
z = (z - z_min)/range
print z

# 21. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (*)
# x = np.random.random((5, 3))*10
# y = np.random.random((3, 2))*10
x = np.ones((5, 3))
y = np.ones((3, 2))
z = np.dot(x, y)
print x
print y
print z

# 22. Given a 1D array, negate all elements which are between 3 and 8,
# in place. (*)
z = np.arange(11)
z[(3 < z) & (z <= 8)] *= -1
print 3 < z
print z <= 8
print z

# 23. Create a 5x5 matrix with row values ranging from 0 to 4 (**)
z = np.zeros((5, 5))
z += np.arange(5)
print np.arange(5)
print z

# 24. Consider a generator function that generates 10 integers and use it
# to build an array (*)
# Reference:
#  http://docs.scipy.org/doc/numpy/reference/generated/numpy.fromiter.html
#  numpy.fromiter(iterable, dtype, count=-1)
#     Create a new 1-dimensional array from an iterable object.
#     Parameters:
#     iterable : iterable object
#         An iterable object providing data for the array.
#
#     dtype : data-type
#         The data-type of the returned array.
#     count : int, optional
#         The number of items to read from iterable. The default is -1,
#           which means all data is read.
#     Returns:
#     out : ndarray
#         The output array.
def generate():
    for x in xrange(10):
        yield x
z = np.fromiter(generate(), dtype=float, count=5)
print z


# 25. Create a vector of size 10 with values from 0 to 1, both
# excluded (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
# Return evenly spaced numbers over a specified interval.
# Returns num evenly spaced samples, calculated over the interval [start, stop].
# The endpoint of the interval can optionally be excluded.
# z = np.linspace(0, 1, 12, endpoint=True)[1:-1]
z = np.linspace(0, 1, 12, endpoint=True)
print z[1:-1]
print z

# 26. Create a random vector and sort it (**)
z = np.random.random(10)
z.sort()
print z

# 27. How to sum a small array faster than np.sum? (**)
z = np.arange(10)
print z
print np.add.reduce(z)

# 28. Consider two random array A and B, check if they are equal (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
#  numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)[source]
#     Returns True if two arrays are element-wise equal within a tolerance.
A = np.random.randint(0, 2, 5)
print A
B = np.random.randint(0, 2, 5)
print B
equal = np.allclose(A, B)
print equal


# 29. Make an array immutable (read-only) (**)
z = np.zeros(10)
z.flags.writeable = False
# z[0] = 1
# ValueError: assignment destination is read-only

# 30. Consider a random 10x2 matrix representing cartesian coordinates,
# convert them to polar coordinates (**)
z = np.random.random((10, 2))
x, y = z[:, 0], z[:, 1]
r = np.sqrt(x**2 + y**2)
t = np.arctan2(y, x)
print x
print y
print r
print t

# 31. Create random vector of size of 10 and replace the maximum value
# by 0 (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
# z = np.random.random(10)
z = np.random.randint(0, 5, 10)
print z
print z.argmax()
z[z.argmax()] = 0  # this line have a bug
print z

# 32. Create a structured array with x and y coordinates covering the
# [0, 1]x[0, 1] area (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
#  numpy.meshgrid(*xi, **kwargs)[source]
#     Return coordinate matrices from coordinate vectors.
#     Make N-D coordinate arrays for vectorized evaluations of N-D
#       scalar/vector fields over N-D grids, given one-dimensional
#       coordinate arrays x1, x2,..., xn.
#     Changed in version 1.9: 1-D and 0-D cases are allowed.
z = np.zeros((10, 10), [('x', float), ('y', float)])
z['x'], z['y'] = np.meshgrid(np.linspace(0, 1, 10),
                             np.linspace(0, 1, 10))
print z

# 33. Given two arrays, x and y, construct the Cauchy matrix C
# (C_{ij} = 1/(x_i - y_j))
x = np.arange(8)
y = x + 0.5
C = 1.0 / np.subtract.outer(x, y)
print C
print(np.linalg.det(C))

# 34. Print the minimum and maximum representable value for each
# numpy scalar type (**)
for dtype in [np.int8, np.int16, np.int32, np.int64]:  #, np.int128
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float16, np.float32, np.float64, np.float128]:  # , np.float80, np.float96, np.float256
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

# 35. How to print all the values of an array? (**)
np.set_printoptions(threshold=np.nan)
# z = np.zeros((250, 2500))
z = np.zeros((25, 25))
print z

# 36. How to find the closest value (to a given scalar) in an array? (**)
# z = np.random.random(10)*10
z = np.arange(100)
v = np.random.uniform(0, 100)
# print z
# print v
# print np.abs(z - v)
index = (np.abs(z - v)).argmin()
print(z[index])

# 37. Create a structured array (or matrix) representing a position (x, y) and
# a color (r, g, b) (**)
z = np.zeros((10, 3), [('position', [('x', float, 1),
                                ('y', float, 1)]),
                  ('color', [('r', float, 1),
                             ('g', float, 1),
                             ('b', float, 1)])
                  ]
             )
print z

# 38. Consider a random vector with shape (100, 2) representing
# coordinates, find point by point distances (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html
#  numpy.atleast_2d(*arys)
#     View inputs as arrays with at least two dimensions.
z = np.random.random((5, 2))
x, y = np.atleast_2d(z[:, 0], np.atleast_2d(z[:, 1]))
d = np.sqrt((x-x.T)**2 + (y-y.T)**2)
print d


# 39. How to convert a float (32 bits) array into an integer (32) in place?
z = np.arange(10, dtype=np.int32)
print z.dtype
z = z.astype(np.float32, copy=False)
print z.dtype

# 40. Consider the following file:
# 1,2,3,4,5
# 6,,,7,8
# ,,9,10,11
# How to read it? (**)
z = np.genfromtxt("missing.dat", delimiter=',')
print z

# 41. What is the equivalent of enumerate for numpy arrays? (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndenumerate.html
#  class numpy.ndenumerate(arr)
#     Multidimensional index iterator.
#     Return an iterator yielding pairs of array coordinates and values.
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndindex.html
#  class numpy.ndindex(*shape)
#     An N-dimensional iterator object to index arrays.
#     Given the shape of an array, an ndindex instance iterates over the
#     N-dimensional index of the array. At each iteration a tuple of indices
#     is returned, the last dimension is iterated over first.
z = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(z):
    print(index, value)
for index in np.ndindex(z.shape):
    print(index, z[index])

# 42. Generate a generic 2D Gaussian-like array (**)
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
d = np.sqrt(x*x, y*y)
print d
sigma, mu = 1.0, 0.0
g = np.exp(-((d - mu)**2/(2.0*sigma**2)))
print g

# 43. How to randomly place p elements in a 2D array? (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
#  numpy.random.choice(a, size=None, replace=True, p=None)
#     Generates a random sample from a given 1-D array
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.put.html
#  numpy.put(a, ind, v, mode='raise')
#     Replaces specified elements of an array with given values.
#     The indexing works on the flattened target array. put is roughly equivalent to:
#       a.flat[ind] = v
n = 10
p = 3
z = np.zeros((n, n), dtype=np.float32)
print z
print np.arange(100)
print np.random.choice(np.arange(100), p, replace=False)
np.put(z, np.random.choice(np.arange(n*n), p, replace=False), 1)
print z

# 44. Subtract the mean of each row of a matrix (**)
x = np.random.rand(5, 6)
print x
# Recent versions of numpy
y = x - x.mean(axis=1, keepdims=True)
# print 'x.mean(axis=1, keepdims=True)'
# print x.mean(axis=1, keepdims=True)
# print 'x.mean(axis=1, keepdims=False)'
# print x.mean(axis=1, keepdims=False)
# print 'x.mean(axis=0, keepdims=True)'
# print x.mean(axis=0, keepdims=True)
# print 'x.mean(axis=0, keepdims=False)'
# print x.mean(axis=0, keepdims=False)
print y
# Older versions of numpy
y = x - x.mean(axis=1).reshape(-1, 1)
print x.mean(axis=1).reshape(-1, 1)
print y

# 45. How to I sort an array by the n-th column? (**)
z = np.random.randint(0, 10, (3, 3))
print z
print z[:, 1]
print z[:, 1].argsort()
print(z[z[:, 1].argsort()])


# 46. How to tell if a given 2D array has null columns?(**)
z = np.random.randint(0, 3, (3, 10))
print z
print z.any(axis=1)
print z.any(axis=0)
print ~z.any(axis=0)
print((~z.any(axis=0)).any())

# 47. Find the nearest value from a given value in an array (**)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flat.html
#  ndarray.flat
#     A 1-D iterator over the array.
#     This is a numpy.flatiter instance, which acts similarly to, but is not
#     a subclass of, Python's built-in iterator object.
z = np.random.uniform(0, 1, 10)
z_ = 0.5
print z
print np.abs(z - z_).argmin()
m = z.flat[np.abs(z - z_).argmin()]
print m

# 48. Consider a given vector, how to add 1 to each element indexed by a
# second vector(be careful with repeated indices)?(***)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html
#  numpy.bincount(x, weights=None, minlength=None)
#     Count number of occurrences of each value in array of
#       non-negative ints.
#     The number of bins (of size 1) is one larger than the largest
#       value in x. If minlength is specified, there will be at least this
#       number of bins in the output array (though it will be longer if
#       necessary, depending on the contents of x). Each bin gives the
#       number of occurrences of its index value in x. If weights is
#       specified the input array is weighted by it, i.e. if a value n
#       is found at position i, out[n] += weight[i] instead of out[n] += 1.
# example:
#   np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
#   array([1, 3, 1, 1, 0, 0, 0, 1])
z = np.ones(10)
i = np.random.randint(0, len(z), 20)
z += np.bincount(i, minlength=len(z))
print i
print np.bincount(i, minlength=len(z))
print z

# 49. How to accumulate elements of a vector X to an array F based on an
# index list I ? (***)
X = [1, 2, 3, 4, 5, 6]
I = [1, 3, 9, 3, 4, 1]
F = np.bincount(I, X)
print X
print I
print np.bincount(I)
print F

w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
x = np.array([0, 1, 1, 2, 2, 2])
print w
print x
print np.bincount(x)
print np.bincount(x, weights=w)

# 50. Considering a (w, h, 3) image of (dtype=ubyte), compute the number
# of unique colors (***)
# w, h = 16, 16
w, h = 4, 4
I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
F = I[..., 0]*256*256 + I[..., 1]*256 + I[..., 2]
print I
print F
n = len(np.unique(F))
print n
print(np.unique(I))

# 51. Considering a four dimensions array, how to get sum over the last
# two axis at once? (***)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
#  ndarray.shape
#     Tuple of array dimensions.
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
#  numpy.reshape(a, newshape, order='C')
#     Gives a new shape to an array without changing its data.
A = np.random.randint(0, 10, (3, 4, 5, 6))
sum = A.reshape(A.shape[: -2] + (-1, )).sum(axis=-1)
print A.shape[: 3]
print A.shape[: 2]
print A.shape[: 1]
print A.shape[: 0]
print A.shape[: -1]
print A.shape[: -2]
print A.shape[: -3]
# print A.shape[: -2]
# print (-1, )
# print A.shape[: -2] + (-1, )
# print A
print sum

# 52. Considering a one-dimensional vector D, how to compute means of
# subsets of D using a vector of same size of describing subset indices?
# (***)
D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums/D_counts
# print D
# print S
# print D_sums
# print D_counts
print D_means

# 53. How to get the diagonal of a dot product? (***)
r = 3
c = 2
A = np.random.random((c, r))
B = np.ones((r, c))
# Slow version
C = np.diag(np.dot(A, B))
# Fast version
C_ = np.sum(A*B.T, axis=1)
# Faster version
_C_ = np.einsum("ij,ji->i", A, B)
print A
print B
print np.dot(A, B)
print 'C'
print C
print C_
print _C_

# 54. Consider the vector [1, 2, 3, 4, 5], how to build a new vector
# with 3 consecutive zeros between each value? (***)
z = np.array([1, 2, 3, 4, 5])
nz = 3
z0 = np.zeros(len(z) + (len(z) - 1)*nz)
z0[::nz+1] = z
print z0[::4]
print z0

# 55. Consider an array of dimension (5, 5, 3), how to multiply it by an
# array with dimensions (5, 5)? (***)
A = 3*np.ones((5, 5, 3))
B = 2*np.ones((5, 5))
print A * B[:, :, None]

# 56. How to swap two rows of an array? (***)
A = np.arange(25).reshape(5, 5)
print A
print A[1]
print A[[1, 0]]
A[[0, 1]] = A[[1, 0]]
print A

# 57. Consider a set of 10 triplets describing 10 triangles (with shared
# vertices), find the set of unique line segments composing all the
# triangles (***)
#  numpy.roll(a, shift, axis=None)
#     Roll array elements along a given axis.
#     Elements that roll beyond the last position are re-introduced at the first.
# >>> x2 = np.reshape(x, (2,5))
# >>> x2
# array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
# >>> np.roll(x2, 1)
# array([[9, 0, 1, 2, 3],
#        [4, 5, 6, 7, 8]])
# >>> np.roll(x2, 1, axis=0)
# array([[5, 6, 7, 8, 9],
#        [0, 1, 2, 3, 4]])
# >>> np.roll(x2, 1, axis=1)
# array([[4, 0, 1, 2, 3],
#        [9, 5, 6, 7, 8]])
#  numpy.repeat(a, repeats, axis=None)[source]
#     Repeat elements of an array.
# >>> x = np.array([[1,2],[3,4]])
# >>> np.repeat(x, 2)
# array([1, 1, 2, 2, 3, 3, 4, 4])
# >>> np.repeat(x, 3, axis=1)
# array([[1, 1, 1, 2, 2, 2],
#        [3, 3, 3, 4, 4, 4]])
# >>> np.repeat(x, [1, 2], axis=0)
# array([[1, 2],
#        [3, 4],
#        [3, 4]])

faces = np.random.randint(0, 100, (10, 3))
print faces
print 'faces.repeat(2, axis=1)'
print faces.repeat(2, axis=1)
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
print 'F = np.roll(faces.repeat(2, axis=1), -1, axis=1)'
print F
F = F.reshape(len(F)*3, 2)
print 'F = F.reshape(len(F)*3, 2)'
print F
F = np.sort(F, axis=1)
G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
G = np.unique(G)
print G

# 58. Given an array C that is a bincount, how to produce an array A
# such that np.bincount(A) == C? (***)
#  numpy.repeat(a, repeats, axis=None)[source]
#     Repeat elements of an array.
# >>> x = np.array([[1,2],[3,4]])
# >>> np.repeat(x, 2)
# array([1, 1, 2, 2, 3, 3, 4, 4])
# >>> np.repeat(x, 3, axis=1)
# array([[1, 1, 1, 2, 2, 2],
#        [3, 3, 3, 4, 4, 4]])
# >>> np.repeat(x, [1, 2], axis=0)
# array([[1, 2],
#        [3, 4],
#        [3, 4]])
_A = [1, 1, 2, 3, 4, 4, 6]
C = np.bincount(_A)
A = np.repeat(np.arange(len(C)), C)
print _A
print C
print A

# 59. How to compute averages using a sliding window over an array? (***) ??
#  numpy.cumsum(a, axis=None, dtype=None, out=None)
#     Return the cumulative sum of the elements along a given axis.
# >>> a = np.array([[1,2,3], [4,5,6]])
# >>> a array([[1, 2, 3],
#        [4, 5, 6]])
# >>> np.cumsum(a) array([ 1,  3,  6, 10, 15, 21])
# >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
# array([  1.,   3.,   6.,  10.,  15.,  21.])
# >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
# array([[1, 2, 3],
#        [5, 7, 9]])
# >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
# array([[ 1,  3,  6],
#        [ 4,  9, 15]])
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
z = np.arange(20)
print z
print moving_average(z, n=3)

# 60. Consider a one-dimensional array Z, build a two-dimensional array
# whose first row is (Z[0], Z[1], Z[2]) subsequent row is shifted by 1
# (last row should be (Z[-3], Z[-2], Z[-1])) (***) ???(not understanding)
from numpy.lib import stride_tricks
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
z = rolling(np.arange(10), 3)
print z

# 61. How to negate a boolean, or to change the sign of a float inplace? (***)
# ??? (not understanding)
z = np.random.randint(0, 2, 100)
print z
np.logical_not(z, out=z)
print z

z = np.random.uniform(-1.0, 1.0, 100)
print z
np.negative(z, out=z)
print z

# 62.

# 63.

# 64.

# 65.

# 66.

# 67.

# 68.

# 69.
