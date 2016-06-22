# Baby Steps - Algebra

# 1. Adding two Scalars

# To get us starated with theano and get a feel of what we are working
# with, let us make a simple functions: add two numbers together. Here is
# how you do it:
import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
print type(x)
print type(y)
print type(z)
# z is yet another Variable which represents the addition of x and y. You
# can use the pp() function to pretty-print out the computation associated
# to z.
from theano import pp
print pp(z)
f = function([x, y], z)
print type(f)
# As a shortcut, you can skip step: f = function([x, y], z), and just use
# a variable's eval() method. the eval() method is not as flexible as
# function() but it can do everything we have in the tutorial so far. It
# has the added benefit of not requiring you to import function(). Here is
# how eval() works:
print numpy.allclose(z.eval({x: 16.3, y:12.1}), 28.4)
# We passed eval() a dictionary mapping symbolic variables to the values to
# the values to substitute for them, and it returned the numerical value of
# the expression.

# And now that we are created our function we can use it:
print f(2, 3)
print numpy.allclose(f(16.3, 12.1), 28.4)


# 2. Adding two Matrices
X = T.dmatrix('X')
Y = T.dmatrix('Y')
Z = X + Y
f = function([X, Y], Z)
print f(numpy.arange(4).reshape((2, 2)),
        numpy.arange(4).reshape((2, 2))*3)
# It is possible to add scalars to matrices, scalars to vectors, etc. The
# behavior of these operations is defined by broadcasting.


# The following types are available:
#     byte: bscalar, bvector, bmatrix, brow, bcol, btensor3, btensor4
#     16-bit integers: wscalar, wvector, wmatrix, wrow, wcol, wtensor3, wtensor4
#     32-bit integers: iscalar, ivector, imatrix, irow, icol, itensor3, itensor4
#     64-bit integers: lscalar, lvector, lmatrix, lrow, lcol, ltensor3, ltensor4
#     float: fscalar, fvector, fmatrix, frow, fcol, ftensor3, ftensor4
#     double: dscalar, dvector, dmatrix, drow, dcol, dtensor3, dtensor4
#     complex: cscalar, cvector, cmatrix, crow, ccol, ctensor3, ctensor4

# 3. Exercise
import theano
a = theano.tensor.vector()  # declare variable
out = a + a**10             # build symbolic expression
f = theano.function([a], out)
print f([0, 1, 2])
# Modify and execute this code to compute this expression: a**2 + b**2 + 2*a*b
a = theano.tensor.fvector()  # declare variable
b = theano.tensor.fvector()  # declare variable
c = a + b
d = c**2
f = theano.function([a, b], d)
print f([0.0, 1.0, 2], [1.0, 3.0, 8])
