# Loop

# Scan
# (1) A general form of recurrence, which can be used for looping.
# (2) Reduction and map (loop over the leading dimensions) are special
# cases of scan.
# (3) You scan a function along some input sequence, producing an output at
# each time-step.
# (4) The function can see the previous K time-steps of your function.
# (5) sum() could be computed by scanning the z + x(i) function over a
# list, given an initial state of z = 0.
# (6) Often a for loop can be expressed as scan() operation, and scan is
# the closest that Theano comes to looping.
# (7) Advantages of using scan over for loops:
#   a. Number of iterations to be part of the symbolic graph.
#   b. Minimizes GPU transfers (if GPU is involved).
#   c. Computes gradients through sequential steps.
#   d. Slightly faster than using a for loop in Python with a compiled
# Theano functionl.
#   e. Cal lower the overall memory usage by detecting the actual amount
# of memory needed.

# The full documentation can be found in the library: Scan.


# 1. Scan Example: Computing tanh(x(t).dot(W) + b) elementwise
import theano
import theano.tensor as T
import numpy as np

# defining the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym),
                               sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=results)

# test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2
print 'x, w, b = '
print x
print w
print b

print compute_elementwise(x, w, b)
print np.tanh(x.dot(w) + b)


# 2. Scan Example: computing the sequence
# x(t) = tanh(x(t - 1).dot(W) + y(t).dot(U) + p(T - t).dot(V))
import theano
import theano.tensor as T
import numpy as np

# define tensor variables
X = T.vector("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

results, updates = theano.scan(lambda y, p, x_tml:
                               T.tanh(T.dot(x_tml, W) +
                                      T.dot(y, U) +
                                      T.dot(p, V)),
                               sequences=[Y, P[::-1]], outputs_info=[X])
compute_seq = theano.function(inputs=[X, W, Y, U, P, V], outputs=results)


# test values
x = np.zeros((2), dtype=theano.config.floatX)
x[1] = 1
w = np.ones((2, 2), dtype=theano.config.floatX)
y = np.ones((5, 2), dtype=theano.config.floatX)
y[0, :] = 3
u = np.ones((2, 2), dtype=theano.config.floatX)
p = np.ones((5, 2), dtype=theano.config.floatX)
p[0, :] = 3
v = np.ones((2, 2), dtype=theano.config.floatX)

print 'x, w, y, u, p, v = '
print x
print w
print y
print u
print p
print v
print 'compute_seq(x, w, y, u, p, v)'
print compute_seq(x, w, y, u, p, v)

# comparison with numpy
x_res = np.zeros((5, 2), dtype=theano.config.floatX)
x_res[0] = np.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
for i in range(1, 5):
    x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4 - i].dot(v))
print 'results from numpy'
print x_res
