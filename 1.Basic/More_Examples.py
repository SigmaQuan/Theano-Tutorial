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
# defined, so the substitutions have to work in any order.

# In practice, a good way of thinking about the givens is as a mechanism
# that allows you to replace any part of your formula with a different
# expression that evaluates to a tensor of same shape and dtype.

# Note: Theano shared variable broadcast pattern default to False for each
# dimensions. Shared varaible size can change over time, so we can't use
# the shape to find the broadcastable pattern. If you want a different
# pattern, just pass it as a parameter theano.shared(...,
# broadcastable=(True, False))


# 5. Copying functions
# Theano functions can be copied, which can be useful for creating similar
# functions but with different shared variables or updates. This is done
# using the copy() method of function objects. The optimized graph of the
# original function is copied, so compilation only needs to be performed
# once.

# Let's start from the accumulator defined above:
state.set_value(0)
print accumulator(10)
print state.get_value()

# We can use copy() to create a similar accumulator but with its own
# internal state using the swap parameter, which is a dictionary of
# shared variables to exchange:
new_state = theano.shared(0)
new_accumulator = accumulator.copy(swap={state: new_state})
print new_accumulator(100)
print new_state.get_value()

# The state of the first function is left untouched:
print state.get_value()

# We now create a copy with updates removed using the delete_updates
# parameter, which is set to False by default:
# null_accumulator = accumulator.copy(delete_updates=True)
# # *** (theano.compile.function_module.UnusedInputError)
#
# # As expected, the shared state is no longer updated:
# print null_accumulator(9000)
# print state.get_value()


# 6. Using Random Numbers

# Because in Theano you first express everything symbolically and
# afterwards compile this expression to get functions, using psesudo-
# random numbers is not as straightforward as it is in NumPy, though
# also not too complicated.

# The way to think about putting randomness into Theano's compuatations
# is to put random variables in your graph. Theano will allocate a NumPy
# RandomStream object (a random number generator) for each such variable,
# and draw from it as necessary. We will call this sort of sequence of
# random objects, so the observations on shared variables hold here as
# well. Theano's random objects are defined and implemented in
# RandomStreams and, at a lower level, in RandomStreamBase.


# 7. Brief Example
# Here's brief example. The setup code is:
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)  # not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
# Here, 'rv_u' represents a random stream of 2x2 matrices of draws from a
# uniform distribution. Likewise, 'rv_n' represents a random stream of 2x2
# matrices of draws from a normal distribution. The distributions that are
# implemented are defined in RandomStreams and, at a lower level, in
# raw_random. They only work on CPU. See 11. Other Implementations for GPU
# version.

# Now let's use these objects. If we call f(), we get random uniform
# numbers. The internal state of the random number generator is
# automatically updated, so we get different random numbers every time.
f_val0 = f()
f_val1 = f()  # different numbers from f_val0
print f_val0
print f_val1

# When we add the extra argument no_default_updates=True to function (as
# in g), then the random number generator state is not affected by calling
# the returned function. So, for example, calling multiple times will
# return the same numbers.
g_val0 = g()  # different numbers from f_val0 anf f_val1
g_val1 = g()  # same numbers as g_val0
print g_val0
print g_val1

# An important remark is that a random variable is drawn at most once
# during any single function execution. So the nearly_zeros function is
# guaranteed to return approximately 0 (except for rounding error) even
# though the rv_u random variable appears three times in the output
# expression. ???(not understanding this paragraph)

# nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)


# 8. Seed Streams

# Random variables can be seeded individually or collectively.

# You can seed just one random variable by seeding or assigning the .rng
# attribute, using the .rng.set_value().
rng_val = rv_u.rng.get_value(borrow=True)  # Get the rng for rv_u
rng_val.seed(89234)                        # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)   # Assign back seeded rng

# You can also seed all the random variables allocated by a RandomStreams
# object by that object's seed method. This seed will be used to seed a
# temporary random number generator, that will in turn generate seeds for
# each of the random variables.
print 'seed'
srng.seed(902340)  # seeds rv_u and rv_n with different seeds each
print f()
print f()
print g()
srng.seed(156456)
print g()


# 9. Sharing Streams Between Functions

# As usual for shared variables, the random number generators used for
# random variables are common between functions. So our nearly_zeros
# function will update the state of the generators used in function f
# above.

# For example:
state_after_v0 = rv_u.rng.get_value().get_state()
print nearly_zeros()
v1 = f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
v2 = f()  # v2 != v1
v3 = f()  # v3 == v1
print v1
print v2
print v3

#
# # 10. Copying Random State Between Theano Graphs
# # ??? (not fully understanding this section)
#
# # In some use cases, a user might want to transfer the 'state' of all
# # random number generators associated with a given theano graph (e.g.
# # g1, with compiled function f1 below) to a second graph (e.g. g2, with
# # function f2). This might arise for example if you are trying to
# # initialize the state of a model, from the parameters of a pickled
# # version of a previous model. For
# # theano.shared_randomstreams.RandomStreams and
# # theano.sandbox.rng_mrg.MRG_RandomStreams this can be achieved by copying
# # elements of the state_updates parameter.
#
# # Each time a random variable is drawn from a RandomStreams object, a tuple
# # is added to the state_updates list. The first element is a shared
# # variable, which represent the state of the random number generator
# # associated with this particular variable, while the second represents the
# # theano graph corresponding to the random number generation process
# # (i.e. RandomFunction{uniform}.0)
# from __future__ import print_function
# import theano
# import numpy
# import theano.tensor as T
# from theano.sandbox.rng_mrg import MRG_RandomStreams
# from theano.tensor.shared_randomstreams import RandomStreams
#
# class Graph():
#     def __init__(self, seed=123):
#         self.rng = RandomStreams(seed)
#         self.y = self.rng.uniform(size=(1, ))
#
# g1 = Graph(seed=123)
# f1 = theano.function([], g1.y)
# g2 = Graph(seed=987)
# f2 = theano.function([], g2.y)
# # By default, the two functions are out of sync.
# f1()
# f2()
#
# def copy_random_state(g1, g2):
#     if isinstance(g1.rng, MRG_RandomStreams):
#         g2.rng.rstate = g1.rng.rstate
#     for (su1, su2) in zip(g1.rng.state.updates, g2.rng.state_updates):
#         su2[0].set_value(su1[0].get_value())
#
# copy_random_state(g1, g2)
# f1()
# f2()


# 11. Other Random Distributions
# omit...

# 12. Other Implementations
# omit...

# 13. A Real Example: Logistic Regression
import numpy
import theano
import theano.tensor as T

rng = numpy.random

N = 400       # training sample size
feats = 784   # number of input variable

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
trainging_steps = 10000

# declare theano symbolic variables
x = T.dmatrix('x')
y = T.dvector('y')

# initialize teh weight vector w randomly
# this and the followint bias variable b are shared so they keep their
# values between training iterations (updates)
w = theano.shared(rng.randn(feats), name='w')

# initialize the bias term
b = theano.shared(0., name='b')

print "Initial model:"
print w.get_value()
print b.get_value()

# construct theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))  # probability that target = 1
prediction = p_1 > 0.5   # the prediction thresholded
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)  # cross-entropy loss
cost = xent.mean() + 0.01 * (w**2).sum()  # this cost to minimize
# compute the gradient of the cost w.r.t weight vector 2 and bias term b
# (we shall return to this following section of this tutorial)
gw, gb = T.grd(cost, [w, b])

# compile
train = theano.function(
    inputs=[x, y],
    outputs=[prediction, xent],
    updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
)
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(trainging_steps):
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value()
print b.get_value()

print "target values for D:"
print D[1]
print "prediction on D:"
print predict(D[0])
