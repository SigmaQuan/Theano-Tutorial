# Conditions

# IfElse v.s. Switch
# (1) Both ops build a condition over symbolic variables.
# (2) IfElse takes a boolean condition and two variables as inputs.
# (3) Swith takes a tensor as condition an two variables as inputs. switch
# is an elementwise operation and is thus more general than ifelse.
# (4) Whereas switch evaluates both output variables, ifelse is lazy and
# only evaluates one variable with respect to the condition.

# Example
from theano import tensor as T
from theano.ifelse import ifelse
import theano, time, numpy

a, b = T.scalars('a', 'b')
x, y = T.matrices('x', 'y')

z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = theano.function([a, b, x, y], z_switch,
                           mode=theano.Mode(linker='vm'))
f_lazyifelse = theano.function([a, b, x, y], z_lazy,
                               mode=theano.Mode(linker='vm'))

val1 = 0.0
val2 = 1.0
big_mat1 = numpy.ones((10000, 20000), dtype=numpy.float32)
big_mat2 = numpy.ones((10000, 20000), dtype=numpy.float32)

n_times = 100

tic = time.clock()
for i in range(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print "time spent evaluating both value %f sec" % (time.clock() - tic)

tic = time.clock()
for i in range(n_times):
    f_lazyifelse(val1, val2, big_mat1, big_mat2)
print "time spent evaluating one value %f sec" % (time.clock() - tic)

# In this example, the ifelse op spends less time (about half as much) than
# switch since it computes only one variable out of the two.

# Unless linker='vm' or linker='cvm' are used, ifelse will compute both
# variables and take the same computation time as switch. Although the
# linker is not currently set by default to cvm, it will be in teh near
# future.
