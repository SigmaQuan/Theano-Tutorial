# Matrix conventions for machine learning
# Rows are horizontal and columns are vertical. Every row is an example.
# Therefore, inputs[10, 5] is a matrix of 10 examples where each
# example has dimension 5. If this would be the input of a neural
# network tehn teh weights from the input to the first hidden layer
# would represent a matrix of size (5, #hid).

# Consider this array:
import numpy as np
w1 = np.asarray([[1., 2], [3, 4], [5, 6]])
print w1
print w1.shape

# This is a 3x2 matrix, i.e. there are 3 row and 2 columns.
# To access the entry in the 3rd row (row #2) and the 1st
# column (column #0):
print w1[2, 0]

# To remember this, keep in mind that we read left-to-right,
# top-to-bottom, so each thing that is contiguous is a row. That is,
# There are 3 rows and 2 columns.

# Broadcasting

# Numpy does broadcasting of arrays of different shapes during arithmetic
# operations. What this means in general is that the smaller array (or
# scalar) is broadcasted across the larger array so that they have
# compatible shapes. The exampel below shows an instances of broadcasting:
a = np.asarray([1.0, 2.0, 3.0])
b = 2.0
print a*b

# The smaller array b (actually a scalar here, which works like
# a 0-d array) in this case is broadcasted to the same size as a during
# the multiplication. This trick is often useful in simplifying how
# expression are written. More detail about broadcasting can be found in
# the numpy user guide.

# http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

# General Broadcasting Rules

# When operating on two arrays, Numpy compares their shapes element-wise.
# It starts with the trailing dimensions, and works its way forward. Two
# dimensions are compatible when
#   1. they are equal, or
#   2. one of them is 1
# if these conditions are not met, a ValueError: frames are not aligned
# excpetion is thrown, indicating that the arrays have imcompatible shapes.
# The size of the resulting array is the maximum size along each dimension
# of the input arrays.

# Arrays do not need to have the same number of dimensions. For example,
# if you have a 256x256x3 array of RGB values, and you want to scale each
# color in the image by a different value, you can multiply the image by
# a one-dimensional array with 3 values. Lining up the sizes of the
# trailing axes of these arrays according to the broadcast rules, shows
# that they are compatible:
#   Image (3d array): 256x256x3
#   Scale (1d array): 3
#   Result (3d array): 256x256x3
# When either of the dimensions compared is one, the other is used. In
# other words, dimensions with size 1 are stretched or 'copied' to match
# other.

# In the following example, both the A and B arrays have axes with lenght
# one that are expanded to a larger size during the broadcast operation:
#   A (4d array): 8x1x6x1
#   B (3d array): 7x1x5
#   Result (4d array): 8x7x6x5

# Here are more examples:
# -----------------------
# A       B       Results
# -----------------------
# 5x4     1       5x4
# 5x4     4       5x4
# 15x3x5  15x1x5  15x3x5
# 15x3x5  3x5     15x3x5
# 15x3x5  3x1     15x3x5
# -----------------------

# Here are examples of shapes that do not broadcast:
# ------------------------------------------------------
# A       B      Reason
# ------------------------------------------------------
# 3       4      Trailing dimensions do not match
# 2x1     8x4x3  Second from last dimensions mismatched
# ------------------------------------------------------

# An example of broadcasting in practice:
x = np.arange(4)
xx = x.reshape(4, 1)
y = np.ones(5)
z = np.ones((3, 4))
print x.shape, y.shape
# print x + y  # ValueError: operands could not be broadcast together
# with shapes (4,) (5,)
print (xx + y).shape
print xx
print y
print xx.shape, y.shape
print xx + y

print x.shape, z.shape
print (x + z).shape
print x
print z
print x + z

# **** outer product
# Broadcasting provides a convenient way of taking the outer product (or
# any other operation) of two ways. The following example shows an outer
# addtion operation of two 1-d arrays:
a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
print a
print a[:, np.newaxis]
print b
print a[:, np.newaxis] + b

# Here the newaxis index operator inserts a new axis into a, making it a
# two-dimensional 4x1 array. Combining the 4x1 array with b, which shape
# (3, ), yields a 4x3 array.
