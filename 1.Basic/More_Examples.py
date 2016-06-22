# More Examples

# At this point it would be wise to begin familiarizing yourself more
# systematically with theano's fundamental objects and operations by
# browsing this section of the library: Basic Tensor Functionality.
# http://deeplearning.net/software/theano/library/tensor/basic.html#libdoc-basic-tensor

# As the tutorial unfold, you should also gradually acquaint yourself with
# the other relevant areas of the library and with relevant subjects of the
# documentation entrance page.


# 1. Logistic Function

# Here's another straightforward example, though a bit more elaborate than
# adding two numbers together. Let's say that you want to compute the
# logistic curve, which is given by:
#   $s_{(x)} = \frac{1}{1 + e^{-x}}$

# You want to compute the function elementwise on matrices of doubles,
# which means that you want to apply this function to each individual
# element of the matrix.

# Well, what you do is this:
import theano
import theano.tensor as T
import numpy as np

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
X = np.arange(20).reshape((5, 4))
print X
print logistic(X)

# It is also the case that:
#   $s_{(x)} = \frac{1}{1 + e^{-x}} = \frac{1 + \tanh(x/2)}{2}$
s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = theano.function([x], s2)
print logistic2(X)


# 2. Computing More than one Thing at the Same Time

# Theano supports functions with multiple outputs. For example, we can
# compute teh element-wise difference, absolute difference, and squared
# difference between two matrices a and b at the same time:
a, b = T.dmatrices('a', 'b')
# Note: dmatrices produces an many outputs ans names you provides. It is
# a shortcut for allocating symbolic variables that we will often use in
# the tutorials
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
dif, dif_abs, dif_squa = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
# print dif, dif_abs, dif_squa
print dif
print dif_abs
print dif_squa


# 3. Setting a Default Value for a Argument

# Let's say you want to define a function that adds two numbers, except
# that you only provide one number, the other input is assumed to be one.
# You can do it like this:
from theano import In
from theano import function
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, In(y, value=1)], z)
print f(33.0)
print f(33, 2)
# This makes use of the In class which allows you to specify properties of
# your function's parameters with greater detail. Here we give a default
# value of 1 for y by creating a In instance with its value field set to 1.

# Inputs with default values must follow inputs without default values
# (like Python's functions). There can be inputs with default values.
# These parameters can be set positionally or by name, as in standard
# Python:
x, y, w = T.dscalars('x', 'y', 'w')
z = w * (x + y)
f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
print f(33)
print f(33, 2)
print f(33, 0, 1)
print f(33, w_by_name=1)
print f(33, w_by_name=1, y=0)
# Note: In does not know the name of the local variables y and w that are
# passed as arguments. The symbolic varaible objects have name attributes
# (set by dscalars in the example above) and these are the names of the
# keyword parameters in the functions taht we build. This is the mechanism
# at work in In(y, value=1). In the case of In(w, value=2, name='w_by_name').
# We override the symbolic variable's name attribute with a name to be used
# for this function.


# 4. Using Shared Variables

# It is also possible to make a function with an internal state. For
# example, let's say we want to make an accumulator: at the beginning,
# the initialized to zero. Then, on each fucntion call, the state is
# incremented by the function's arguments.

# First let's define teh accumulator function. It adds its argument to the
# interanl state, and returns teh old state value.
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
# This code introduces a few concepts. The shared function constructs so-
# called shared variables. These are hybrid symbolic and non-symbolic
# variables whose value may be shared between multiple functions. Shared
# variables can be used in symbolic expressions just like the objects
# returned by dmatrices(...) but they also have an internal value that
# defines the value taken by this symbolic variable in all the functions
# that use it. It is called a shared variable because its value is shared
# between many .set_value() methods. We will com back to this soon.

# The other new thing in this code is the updates parameter of function.
# updates must be supplied with a list of pairs of the form (shared-
# variable, new expression). It can also be a dictionary whose keys are
# shared-variables and values are the new expression. Either way, it means
# "whenever this function runs, it will replace the .value of each shared
# variable with the result fo the corresponding expression". Above, our
# accumulator replaces the state's value with the sum of the state and the
# increment amount.

# Let's try it out!
print state.get_value()
print accumulator(1)
print state.get_value()
print accumulator(300)
print state.get_value()
# It is possible to reset the state. Just use the .set_value() method:
state.set_value(-1)
print accumulator(3)
print state.get_value()

# As we mentioned above, you can define more than one function to use
# the same shared variable. These functions call all update the value.
decrementor = function([inc], state, updates=[(state, state-inc)])
print decrementor(3)
print state.get_value()

# You might be wondering why the updates mechanism exist. You can always
# achieve a similar result by returning the new expressions, and working
# with them in NumPy as usual. The udpates mechanism can be a syntactic
# convenience, but it is mainly there for in-place algorithms (e.g. low-
# rank matrix updates). Also, theano has more control over where and how
# shared variables are allocated, which is one of the important elements
# of getting good performance on the GPU.

# It may happen that you expressed some formula using a shared variable,
# but you do not want to use its value. In this case, you can use the
# givens parameter of function which replaces a particular node in a graph
# for the purpose of one particular function.
fn_of_state = state * 2 + inc
# The type of foo must match the shared variable we are replacing with
# the 'givens
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)  # we're using 3 for the state, not state. value
print state.get_value()
# The givens parameter can be used to replace any symbolic variable, not
# just a shared variable. You can replace constants, and expressions, in
# general. Be careful though, not to allow the expression introduced by a
# givens substitution to be co-dependent, the order of substitution is not




# 5. Copying functions


# 6. Using Random Numbers


# 7. Brief Example


# 8. Seed Streams


# 9. Sharing Streams Between Functions


# 10. Copying Random State Between Theano Graphs


# 11. Other Random Distributions


# 12. Other Implementations


# 13. A Real Example: Logistic Regression