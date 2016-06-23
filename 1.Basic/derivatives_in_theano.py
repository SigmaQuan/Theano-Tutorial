# Derivatives in Theano


# 1. Computing Gradients

# Now let's use Theano for a slightly more sophisticated task: create
# a function which computes the derivative of some expression y with
# respected to its parameters x. To do this we will use the macro T.grad.
# For instance, we can compute the gradient of $x^2$ with respect to x.
# Note that: $d(x^2)/dx = 2x$.

# Here is the code to compute this gradient:
import numpy
import theano
import theano.tensor as T
from theano import pp

x = T.dscalar('x')
y = x**2
gy = T.grad(y, x)
print pp(gy)  # print out the gradient prior to optimization
f = theano.function([x], gy)
print f(4)
print numpy.allclose(f(94.2), 188.4)
# In this example, we can see from pp(gy) that we are computing the correct
# symbolic gradient. fill((x**2), 1.0) means to make a matrix of the same
# shape as $x**2$ and fill it with 1.0.

# Note: The optimizer simplifies the symbolic gradient. You can see this
# by digging inside the internal properties to the compiled function.
print pp(f.maker.fgraph.outputs[0])
# (TensorConstant{2.0} * x)
# After optimization there is only one Apply node left in the graph, which
# doubles the input.

# We can also compute the gradient of complex expressions such as the
# logistic function defined above. It turns out that the derivative of the
# logistic is:
# $ds(x)/dx = s(x)(1 - s(x))$
x = T.dmatrix('x')
s = T.sum(1 / (1 + T.exp(-x)))
gs = T.grad(s, x)
d_logistic = theano.function([x], gs)
print d_logistic([[0, 1], [-1, -2]])

# In general, for any scalar expression s, T.grad(s, w) provides the
# Theano expression for computing $\frac{\partial s}{\partial w}$. In this
# way Theano can be used for doing efficient symbolic differentiation (as
# the expression returned by T.grad() will be optimized during
# compilation), even for function with many inputs. (see automatic
# differentiation for a description of symbolic differentiation).

# 2. Computing the Jacobian


# 3. Computing the Hessian


# 4. Jacobian times a Vector


# 5. R-operator


# 6. L-operator


# 7. Final Pointers
