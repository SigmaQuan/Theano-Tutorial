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


# 3. Scan Example: computing norms of lines of X
import theano
import theano.tensor as T
import numpy as np

# define tensor variable
X = T.matrix('X')
results, updates = theano.scan(lambda x_i: T.sqrt((x_i**2).sum()),
                               sequences=[X])
compute_norm_lines = theano.function(inputs=[X], outputs=results)

# test value
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
print 'x'
print x
print "compute_norm_lines(x)"
print compute_norm_lines(x)

# comparison with numpy
print "comparison with numpy"
print np.sqrt((x**2).sum(1))


# 4. Scan Example: computing norms of columns of X
import theano
import theano.tensor as T
import numpy as np

# define tensor variable
X = T.matrix('X')
results, updates = theano.scan(lambda x_i: T.sqrt((x_i**2).sum()),
                               sequences=[X.T])
compute_norm_cols = theano.function(inputs=[X], outputs=results)

# test value
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
print "compute_norm_cols(x)"
print compute_norm_cols(x)

# comparison with numpy
print "comparison with numpy"
print np.sqrt((x**2).sum(0))


# 5. Scan Example: computing trace of X
import theano
import theano.tensor as T
import numpy as np
floatX = "float32"

# define tensor variable
X = T.matrix("X")
results, updates = theano.scan(
    lambda i, j, t_f: T.cast(X[i, j] + t_f, floatX),
    sequences=[T.arange(X.shape[0]), T.arange(X.shape[1])],
    outputs_info=np.asarray(0., dtype=floatX))
result = results[-1]
compute_trace = theano.function(inputs=[X], outputs=result)

# test value
x = np.eye(5, dtype=theano.config.floatX)
x[0] = np.arange(5, dtype=theano.config.floatX)
print "compute_trace(x)"
print compute_trace(x)

# cmparison with numpy
print "cmparison with numpy"
print np.diagonal(x).sum()


# 6. Scan Example: computing the sequence
# x(t) = x(t - 2).dot(U) + x(t - 1).dot(V) + tanh(x(t - 1).dot(W) + b)
import theano
import theano.tensor as T
import numpy as np

# define tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
V = T.matrix("V")
n_sym = T.iscalar("n_sym")

results, updates = theano.scan(
    lambda x_tm2, x_tm1:
    T.dot(x_tm2, U) + T.dot(x_tm1, V) + T.tanh(T.dot(x_tm1, W) + b_sym),
    n_steps=n_sym, outputs_info=[dict(initial=X, taps=[-2, -1])])
compute_seq2 = theano.function(inputs=[X, U, V, W, b_sym, n_sym],
                               outputs=results)

# test values
x = np.zeros((2, 2), dtype=theano.config.floatX)  # the initial value must
# be able to return x[-2]
x[1, 1] = 1
w = 0.5 * np.ones((2, 2), dtype=theano.config.floatX)
u = 0.5 * (np.ones((2, 2), dtype=theano.config.floatX) -
           np.eye(2, dtype=theano.config.floatX))
v = 0.5 * np.ones((2, 2), dtype=theano.config.floatX)
n = 10
b = np.ones((2), dtype=theano.config.floatX)
print "compute_seq2(x, u, v, w, b, n)"
print compute_seq2(x, u, v, w, b, n)

# comparion with numpy
print "comparion with numpy"
x_res = np.zeros((10, 2))
x_res[0] = x[0].dot(u) + x[1].dot(v) + np.tanh(x[1].dot(w) + b)
x_res[1] = x[1].dot(u) + x_res[0].dot(v) + np.tanh(x_res[0].dot(w) + b)
x_res[2] = x_res[0].dot(u) + x_res[1].dot(v) + np.tanh(x_res[1].dot(w) + b)
for i in range(2, 10):
    x_res[i] = (x_res[i - 2].dot(u) +
                x_res[i - 1].dot(v) +
                np.tanh(x_res[i - 1].dot(w) + b))
print x_res


# 7. Scan Example: computing the Jacobian of y = tanh(v.dot(A)) wrt v
import theano
import theano.tensor as T
import numpy as np

# define tensor variable
v = T.vector()
A = T.matrix()
y = T.tanh(T.dot(v, A))
results, updates = theano.scan(
    lambda i: T.grad(y[i], v),
    sequences=[T.arange(y.shape[0])])
compute_jac_t = theano.function(
    [A, v], results,
    allow_input_downcast=True)  # shape(d_out, d_in)

# test values
x = np.eye(5, dtype=theano.config.floatX)[0]
w = np.eye(5, 3, dtype=theano.config.floatX)
w[2] = np.ones((3), dtype=theano.config.floatX)
print "compute_jac_t(w, x)"
print compute_jac_t(w, x)

# compare with numpy
print ((1 - np.tanh(x.dot(w))**2) * w).T
# Note that we need to iterate over the indices of y and not over the
# elements of y. The reason is that scan create a placeholder variable
# for its internal function and this placeholder variable does not have
# the same dependencies than the variables that will replace it.


# 8. Scan Example: accumalate number of loop during a scan
import theano
import theano.tensor as T
import numpy as np

# define shared variables
k = theano.shared(0)
n_sym = T.iscalar("n_sym")

results, updates = theano.scan(lambda: {k: (k + 1)}, n_steps=n_sym)
accumulator = theano.function([n_sym], [], updates=updates,
                              allow_input_downcast=True)
print k.get_value()
accumulator(5)
print k.get_value()


# 9. Scan Example: computing tanh(v.dot(W) + b) * d where d is binomial
import theano
import theano.tensor as T
import numpy as np

# define tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

# define shared random stream
trng = T.shared_randomstreams.RandomStreams(1234)
d = trng.binomial(size=W[1].shape)

results, updates = theano.scan(
    lambda v: T.tanh(T.dot(v, W) + b_sym) * d,
    sequences=X)
compute_with_bnoise = theano.function(
    inputs=[X, W, b_sym],
    outputs=results,
    updates=updates,
    allow_input_downcast=True)

# test
x = np.eye(10, 2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)

print compute_with_bnoise(x, w, b)


# 10. Scan Example: Computing pow(A, k)
import theano
import theano.tensor as T
theano.config.warn.subtensor_merge_bug = False

k = T.iscalar("k")
A = T.vector("A")

def inner_fct(prior_result, B):
    return prior_result * B

# Symbolic description of the result
result, updates = theano.scan(fn=inner_fct,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)
# Scan has provided us with A ** 1 trough A ** k. Keep only the least
# value. Scan notices this and does not waste memory saving them.
final_result = result[-1]

power = theano.function(inputs=[A, k],
                        outputs=final_result,
                        updates=updates)

print power(range(10), 5)


# 11. Scan Example: calculating a Polynomial
import numpy
import theano
import theano.tensor as T
theano.config.warn.subtensor_merge_bug = False

coeficients = theano.tensor.vector("coefficients")
x = T.scalar("x")
max_coefficients_supported = 10000

# Genearte the components of the polynomial
full_range = theano.tensor.arange(max_coefficients_supported)
components, updates = theano.scan(
    fn=lambda coeff, power, free_var: coeff * (free_var ** power),
    outputs_info=None,
    sequences=[coeficients, full_range],
    non_sequences=x
)
polynomial = components.sum()
calculate_polynomial = theano.function(
    inputs=[coeficients, x],
    outputs=polynomial
)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)

print calculate_polynomial(test_coeff, 3)


# 12. Exercise

# Run both examples.

# Modify and execute teh polynomial example to have the reduction done by
# scan.
# from __future__ import print_function
import numpy

import theano
import theano.tensor as tt
from six.moves import xrange

# 3. Reduction performed inside scan
theano.config.warn.subtensor_merge_bug = False
coefficients = tt.vector("coefficients")
max_coefficients_supported = 1000

# Generate the components of the polynomial
full_range = tt.arange(max_coefficients_supported)

outputs_info = tt.as_tensor_variable(numpy.asarray(0, 'float64'))

components, updates = theano.scan(
    fn=lambda coeff, power, prior_value, free_var:
    prior_value + (coeff*(free_var**power)),
    sequences=[coefficients, full_range],
    outputs_info=outputs_info,
    non_sequences=x
)

polynomial = components[-1]
calculate_polynomial = theano.function(
    inputs=[coefficients, x],
    outputs=polynomial,
    updates=updates
)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)

print(calculate_polynomial(test_coeff, 3))

