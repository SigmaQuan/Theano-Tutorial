# Baby Steps - Algebra

# Adding two Scalars

# To get us starated with theano and get a feel of what we are working
# with, let us make a simple functions: add two numbers together. Here is
# how you do it:
import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

# And now that we are created our function we can use it:
print f(2, 3)
print numpy.allclose(f(16.3, 12.1), 28.4)
