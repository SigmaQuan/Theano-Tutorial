# How Shape Information is Handled by Theano

# It is not possible to strictly enforce the shape of a Theano variable
# when building a graph since the particular value provided at run-time
# for a parameter of a Theano function may condition the shape of the
# Theano variables in its graph.

# Currently, information regarding shape is used in two ways in Theano:
# 1. To generate faster C code for the 2d convoution on the CPU and the
# GPU, when the exact output shape is known in advance.
# 2. To remove computations in the graph when we only want to know the
# shape, but not the actual value of a variable. This is done with the
# Op.infer_shape method.

# Example:
import theano
x = theano.tensor.matrix("x")
f = theano.function([x], (x ** 2).shape)
theano.printing.debugprint(f)
# MakeVector{dtype='int64'} [id A] ''   2
#  |Shape_i{0} [id B] ''   1
#  | |x [id C]
#  |Shape_i{1} [id D] ''   0
#    |x [id C]
# The output of this compiled function does not contain any multiplication
# or power. Theano has removed them to compute directly the shape of the
# output.


# 1. Shape Inference Problem
# Theano propagates information about shape in the graph. Sometimes this
# can lead to errors. Consider this example:
import numpy
import theano
x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')
z = theano.tensor.join(0, x, y)
xv = numpy.random.rand(5, 4)
yv = numpy.random.rand(3, 3)
f = theano.function([x, y], z.shape)
theano.printing.debugprint(f)
# MakeVector{dtype='int64'} [id A] ''   4
#  |Elemwise{Add}[(0, 0)] [id B] ''   3
#  | |Shape_i{0} [id C] ''   1
#  | | |x [id D]
#  | |Shape_i{0} [id E] ''   2
#  |   |y [id F]
#  |Shape_i{1} [id G] ''   0
#    |x [id D]

# f(xv, yv)  # Does not raise an error as should be.
# TypeError: ('Bad input argument to theano function with name
#  "/home/qzb/Documents/Research/PythonProjects/Theano-Tutorial
#   /1.Basic/how_shape_information.py:40"  at index 0(0-based)',
#   'TensorType(float32, matrix) cannot store a value of dtype float64
#   without risking loss of precision. If you do not mind this loss,
#   you can: 1) explicitly cast your data to float32, or 2) set
#   "allow_input_downcast=True" when calling "function".',
#   array([[ 0.69066755,  0.01642843,  0.89755138,  0.35554418],
#        [ 0.4615057 ,  0.79271436,  0.7792367 ,  0.3228614 ],
#        [ 0.35713128,  0.97761754,  0.0074782 ,  0.51112224],
#        [ 0.30555429,  0.84245706,  0.39527505,  0.02838553],
#        [ 0.67286326,  0.48256939,  0.85513945,  0.33757078]]))

f = theano.function([x, y], z)  # Do not take the shape.
theano.printing.debugprint(f)
# HostFromGpu [id A] ''   3
#  |GpuJoin [id B] ''   2
#    |TensorConstant{0} [id C]
#    |GpuFromHost [id D] ''   1
#    | |x [id E]
#    |GpuFromHost [id F] ''   0
#      |y [id G]

# As you can see, when asking only for the shape of some computation (join
# in the example), ans inferred shape is computed directly, without
# executing the the computation itself (there is no join in the first output
# or debugprint).

# This makes the computation of the shape faster, but it can also hide
# errors. In this example, the computation of the shape of the output of
# join is done only based on the first input Theano variable, which leads
# to an error.

# This might happen with other ops such as elemwise and dot, for example.
# Indeed, to perform some optimizations (for speed or stability, for
# instance), Theano assumes that the computation is correct and consistent
# in the first place, as it does here.

# You can detect those problems by running the code without this
# optimization, using the Theano flag
#   optimizer_excluding=local_shape_to_shape_i.
# You can also obtain the same effect by running n the modes FAST_COMPILE
# (it will not apply this optimization, nor most other optimizations) or
# DebugMode (it will test before and after all optimizations (mush
# slower)).


# 2. Specifing Exact Shape
# Currently, specifying a shape is not as easy and flexible as we wish and
# we plan some upgrade. Here is the current state of what can be done:
#   (1) You can pass the shape info directly to the ConvOp created when calling
#   conv2d. You simply set the parameters image_shape and filter_shape
#   inside the call. They must be tuples of 4 elements. For ample:
#       theano.tensor.nnet.conv2d(...,
#                                 image_shape=(7, 3, 5, 5),
#                                 filter_shape=(2, 3, 4, 4))
#   (2) You can use teh SpecifyShape op to add shape information anywhere
#   in the graph. This allows to perform some optimizations. In the
#   following example, this make it possible to precompute the Theano
#   function to a constant.
import theano
x = theano.tensor.matrix()
x_specify_shape = theano.tensor.specify_shape(x, (2, 2))
f = theano.function([x], (x_specify_shape ** 2).shape)
theano.printing.debugprint(f)
# DeepCopyOp [id A] ''   0
#  |TensorConstant{(2,) of 2} [id B]


# 3. Future Plans
# The parameter "constant shape" will be added to theano.shared(). This is
# probably the most frequent occurrence with shared varaibles. It will make
# the code simpler and will make the code simpler and will make it possible
# to check that the shape does not change when updating the shared varaible.