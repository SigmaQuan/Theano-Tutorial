# Using the GPU

# For an introductory discussion of Graphical Processing Uints and their
# use for intensive parallel computation purposes, see GPGPU.
# http://en.wikipedia.org/wiki/GPGPU

# One of Theano's design goals is to specify compuations at an abstract
# level, so that the internal function compiler has a lot of flexibility
# about how to carry out those computations on a graphics card.

# There are two ways currently to use a gpu, one of which only supports
# NVIDIA cards and the other, in development, that should support any
# OpenCL device as well as NVIDIA cards(GPUArray Backend).

# 1. CUDA backend
# If you have not done so already, you will need to install Nvidia's
# GPU-programming toolchain (CUDA) and configure Theano to use it. We
# provide intallation instructions for Linux, MacOS and Windows.
#
# # 1.1 Testing Theano with GPU
# # To see if your GPU is being used, cut and paste the following grogramm
# # into a file and run it.
# from theano import function, config, shared, sandbox
# import theano.tensor as T
# import numpy
# import time
#
# vlen = 10 * 30 * 768  # 10 x cores x threads per core
# iters = 1000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], T.exp(x))
# print f.maker.fgraph.toposort()
# t_begin = time.time()
# for i in range(iters):
#     r = f()
# t_end = time.time()
#
# print "Looping %d times took %f seconds" % (iters, t_end-t_begin)
# print "Result is %s" % (r,)
#
# if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
#     print "Used the cpu"
# else:
#     print "Used the gpu"
# # The program just computes the exp() of a bunch of random numbers. Note
# # that we use the shared function to make sure that input x is stored on
# # the graphics device.
#
# # If I run this program (in using_gpu.py) with device=cpu, my computer
# # takes a little over 3 seconds, whereas on the GPU it takes over 4
# # seconds, whereas on the GPU it takes just over 0.32 seconds. The GPU will
# # not always produce the exact same floating-point numbers as the CPU. As a
# # benchmark, a loop that calls numpy.exp(x.get_value()) takes about more
# # seconds.
#
# # THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python using_gpu.py
# # [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
# # Looping 1000 times took 4.728252 seconds
# # Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
# #   1.62323284]
# # Used the cpu
#
# # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu.py
# # Using gpu device 0: GeForce GTX TITAN (CNMeM is enabled with initial size: 85.0% of memory, CuDNN 4007)
# # [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
# # Looping 1000 times took 0.324269 seconds
# # Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
# #   1.62323296]
# # Used the gpu
#
# # THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python using_gpu.py
# # Using gpu device 1: GeForce GTX TITAN (CNMeM is enabled with initial size: 85.0% of memory, CuDNN 4007)
# # [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
# # Looping 1000 times took 0.323090 seconds
# # Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
# #   1.62323296]
# # Used the gpu
#
# # Note that GPU oeprations in Theano require for now floatX to be float 32
# # (see also below).

# 1.2 Returning a Handle to Device-Allocated Data
# The speedup is not greater in the preceding example because the function
# is returning its result as a Numpy ndarray which has already been copied
# from the device to the host for your convenience. This is what makes it
# so easy to swap in device=gpu, but if you don't mind less portability,
# you might gain a bigger speedup by changing the graph to express a
# computation with a GPU-stored result. The gpu_from_host op means "copy
# the input from the host to the GPU" and it is optimized away after the
# T.exp(x) is replaced by a GPU version of exp().

# from theano import function, config, shared, sandbox
# import theano.tensor as T
# import numpy
# import time
#
# vlen = 10 * 30 * 768  # 10 x cores x threads per core
# iters = 100000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
# print f.maker.fgraph.toposort()
# t_begin = time.time()
# for i in range(iters):
#     r = f()
# t_end = time.time()
#
# print "Looping %d times took %f seconds" % (iters, t_end-t_begin)
# print "Result is %s" % (r,)
#
# if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
#     print "Used the cpu"
# else:
#     print "Used the gpu"

# The output from this program is:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu.py
# Using gpu device 0: GeForce GTX TITAN (CNMeM is enabled with initial size: 85.0% of memory, CuDNN 4007)
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
# Looping 1000 times took 0.015918 seconds
# Result is CudaNdarray([ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
#   1.62323296])
# Used the gpu

# Here we've shaved of about 95% (1-0.015918/0.323090) of the run-time by
# simply not copying the resulting array back to the host. The object
# returned by each function call is now not a Numpy array but a
# CudaNdarray which can be converted to a NumPy ndarray by the normal
# Numpy casting mechanism using someting like numpy.asarray().

# For even more speed you can play with the borrow flag. See Borrowing when
# Constructing Function Objects.
# http://deeplearning.net/software/theano/tutorial/aliasing.html#borrowfunction

# 1.3 (*****) What Can be Accelerated on the GPU
# The performance characteristics will change as we continue to optimize
# our implementations, and vary from device to device, but to give a rough
# idea of what do expect right now:
# (1) Only computations with float32 data-type can be accelerated. Better
#     support for float64 is expected in upcoming hardware but float64
#     computations are still relatively slow (Jan 2010).
# (2) Matrix multiplication, convolution, and large element-wise operations
#     can be accelerated a lot (5-50) when arguments are large enough to
#     keep 30 processors busy.
# (3) Indexing, dimension-shuffling and constant-time reshaping will be
#     equally fast on GPU and CPU.
# (4) Summation over rows/columns of tensors can be a little slower on the
#     GPU than on the GPU.
# (5) Copying of large quantities of data to and and from a devices is
#     relatively slow, and often cancels most of the advantage of one or
#     two accelerated functions on that data. Getting GPU performance
#     largely hinges on making data transfer to the device pay off.

# 1.4 (*****) Tips for Improving Performance on GPU
# (1) Consider adding floatX=float32 to your .theanorc file if you plan to
#     do a lot of GPU work.
# (2) Use the Theano flag allow_gc=False. See GPU Async Capabilities.
# (3) Prefer constructs like matrix, vector and scalar to dmatrix, dvector
#     and dscalar because the former will give you float32 variable when
#     floatX=float32.
# (4) Ensure that your output variables have a float32 dtype and not
#     float64. The more float32 variable are in your graph, the more work
#     the GPU can do for you.
# (5) Minimize transfers to the GPU device, by using shared float32
#     variables to store frequently-accessed data (see shared()). When
#     using the GPU, float32 tensor shared variables are stored on the
#     GPU by default to eliminate transfer time for GPU ops using those
#     variables.
# (6) If you aren't happy with the performance you see, try running your
#     script with profile=True flag. This should print some timing
#     information at program termination. Is time being used sensibly? If
#     an op or Apply is taking more time than its share, then if you know
#     something about GPU programming, have a look at how it's implemented
#     in theano.sandbox.cuda. Check the line similar to Spent Xs(X%) in cpu
#     op, Xs(X%) in gpu op and Xs(X%) in transfer op. This can tell you if
#     If not enough of your graph is on the GPU or if there is too much
#     memory transfer.
# (7) Using nvcc options. nvcc supports those options to speed up some
#     computations: -ftz=true to flush denormals values to zeros.
#     -prec-div=false and -prec-sqrt=false options to speed up division
#     and square root operation by being less precise. You can enable all
#     of them with the nvcc. flags=-use_fast_math Theano flag or you can
#     enable them individually as in this example: nvcc. flags=-ftz=true
#     -prec-div=false.
# (8) To investigate whether if all the Ops int he computational graph are
#     running on GPU. It is possible to debug or check your code by
#     providing a value to assert_no_cpu_op flag, i.e. warn, for warning
#     raise for raising an error or pdb for putting a breakpoint in the
#     computational graph if there is a CPU Op.

# 1.5 GPU Async Capabilities
# Ever since Theano 0.6 we started to use the asynchronous capability of
# GPUs. This allows us to be faster but with the possibility that some
# errors may be raised later than when they should occur. This can cause
# difficulties when profiling Theano apply nodes. There is a NVIDIA driver
# feature to help with these issues. If you set the environment variable
# CUDA_LAUNCH_BLOCKING=1 then all kernel call will be automatically
# synchronized. This reduces performance but provides good profiling and
# appropriately placed error messages.

# This feature interacts with Theano garbage collection of intermediate
# results. To get the most of this feature, you need to disable the gc as
# it inserts synchronization points int he graph. Set the Theano flag
# allow_gc=False to get even faster speed! This will raise the memory
# usage.

# 1.6 Changing the Value of Shared Variables
# To change the value of a shared variable, e.g. to provide new data to
# processes, use shared_variable.set_value(new_value). For a lot more
# about this, see Understanding Memory Aliasing for Speed and Correctness.
# http://deeplearning.net/software/theano/tutorial/aliasing.html#aliasing

#
# # 1.6.1 Exercise
# # Consider again the logistic regression:
# import numpy
# import theano
# import theano.tensor as T
#
# rng = numpy.random
#
# N = 400
# feats = 784
# D = (rng.randn(N, feats).astype(theano.config.floatX),
#      rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
# training_steps = 10000
#
# # Declare Theano symbolic variables
# x = T.matrix('x')
# y = T.vector('y')
# w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name='w')
# b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name='b')
# x.tag.test_value = D[0]
# y.tag.test_value = D[1]
#
# # Construct theano expression graph
# p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))  # probability of having a one
# prediction = p_1 > 0.5  # the prediction that is done: 0 or 1
# xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
# cost = xent.mean() + 0.01 * (w**2).sum()  # the cost to optimize
# gw, gb = T.grad(cost, [w, b])
#
# # Compile expressions to functions
# train = theano.function(
#     inputs=[x, y],
#     outputs=[prediction, xent],
#     updates=[(w, w - 0.01 * gw), (b, b - 0.01 * gb)],
#     name='train'
# )
# predict = theano.function(
#     inputs=[x],
#     outputs=prediction,
#     name='predict'
# )
#
# if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm']
#         for x in train.maker.fgraph.toposort()]):
#     print "Used the cpu"
# elif any([x.op.__class__.__name__ in ['GpuGemv', 'GpuGemv']
#         for x in train.maker.fgraph.toposort()]):
#     print "Used the gpu"
# else:
#     print "ERROR, not able to tell if theano used the cpu or the gpu"
#
# for i in range(training_steps):
#     pred, err = train(D[0], D[1])
#
# print "target value for D"
# print D[1]
#
# print "predition on D"
# print predict(D[0])


# 2. GpuArray Backend
# If you have not done so already, you will need to install libgpuarray as
# well as least one computing toolkit. Instructions for doing so are
# provided at libgpuarray.

# While all types of devices are supported if using OpenCL, for the
# remainder of this section, whatever compute device you are using will be
# referred to as GPU.

# Warning:
# The backend was designed to support OpenCL, however current support is
# incomplete. A lot of very useful ops still do not support it because they
# were ported from the old backend with minimal change.

# 2.1 Testing Theano with GPU

# Too see if your GPU is being used, cut and paste the following program
# into a file and run it.

# Use teh Theano flag device=cuda to require the use of the GPU. Use the
# flag device=cuda{0,1,...) to specify which GPU to use.

# from theano import function, config, shared, tensor
# import numpy
# import time
#
# vlen = 10 * 30 * 768  # 10 x cores x threads per core
# iters = 1000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], tensor.exp(x))
# print f.maker.fgraph.toposort()
# t_begin = time.time()
# for i in range(iters):
#     r = f()
# t_end = time.time()
# print "Looping %d times took %f seconds" % (iters, t_end - t_begin)
#
# if numpy.any([isinstance(x.op, tensor.Elemwise) and
#                      ('Gpu' not in type(x.op).__name__)
#              for x in f.maker.fgraph.toposort()]):
#     print "Used the cpu"
# else:
#     print "Used the gpu"

# The program just computes exp() of a bunch of random numbers. Note that
# we use the theano.shared() function to make sure that the input x is
# stored on the GPU.

# THEANO_FLAGS=device=cpu python using_gpu.py
# [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
# Looping 1000 times took 4.691053 seconds
# Used the cpu

# THEANO_FLAGS=device=gpu python using_gpu.py
# Using gpu device 0: GeForce GTX TITAN (CNMeM is enabled with initial size: 85.0% of memory, CuDNN 4007)
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
# Looping 1000 times took 0.335471 seconds
# Used the gpu

# 2.2 Returning a Handle to Device-Allocated Data
# By default functions that execute on the GPU still return a standard
# numpy ndarray. A transfer operation is inserted just before the results
# are returned to ensure a consistent interface with GPU code. This allows
# changing the device some code runs on by only replacing the value of the
# device flag without touching the code.

# If you don't mind a loss of flexibility, you can ask theano to return the
# GPU object directly. The following code is modified to do just that.

from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x cores x threads per core
iters = 1000

rng = numpy.random.RandomState(2)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x).transfer('cpu'))
print f.maker.fgraph.toposort()
t_begin = time.time()
for i in range(iters):
    r = f()
t_end = time.time()
print "Looping %d times took %f seconds" % (iters, t_end - t_begin)
print "Result is %s" % (numpy.asarray(r),)
if numpy.any([isinstance(x.op, tensor.Elemwise) and
                      ('Gpu' not in type(x.op).__name__)
              for x  in f.maker.fgraph.toposort()]):
    print "Used the cpu"
else:
    print "Used the gpu"

# Here tensor.exp(x).transfer(None) means "copy exp(x) to the GPU", with
# None the default GPU context when not explicitly given. For information
# on how to set GPU contexts, see Using multiple GPUs.

# The output is:

# f = function([], tensor.exp(x).transfer('gpu'))
# THEANO_FLAGS=device=gpu python using_gpu.py
# Using gpu device 0: GeForce GTX TITAN (CNMeM is enabled with initial size: 85.0% of memory, CuDNN 4007)
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
# Looping 1000 times took 0.015636 seconds
# Result is [ 1.5465008   1.02626526  1.73266816 ...,  2.26791716  1.88959861
#   1.06480491]
# Used the gpu

# f = function([], tensor.exp(x).transfer('cpu'))
# root@GPUSE:/home/qzb/Documents/Research/PythonProjects/Theano-Tutorial/2.Advanced# THEANO_FLAGS=device=gpu python using_gpu.py
# Using gpu device 0: GeForce GTX TITAN (CNMeM is enabled with initial size: 85.0% of memory, CuDNN 4007)
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
# Looping 1000 times took 0.371334 seconds
# Result is [ 1.5465008   1.02626526  1.73266816 ...,  2.26791716  1.88959861
#   1.06480491]
# Used the gpu

# f = function([], tensor.exp(x).transfer('gpu'))
# THEANO_FLAGS=device=cuda0 python using_gpu.py
# ERROR (theano.sandbox.gpuarray): pygpu was configured but could not be imported
# Traceback (most recent call last):
#   File "/root/Theano/theano/sandbox/gpuarray/__init__.py", line 21, in <module>
#     import pygpu
# ImportError: No module named pygpu
# Traceback (most recent call last):
#   File "using_gpu.py", line 358, in <module>
#     f = function([], tensor.exp(x).transfer('gpu'))
#   File "/root/Theano/theano/tensor/var.py", line 385, in transfer
#     return theano.tensor.transfer(self, target)
#   File "/root/Theano/theano/tensor/basic.py", line 2890, in transfer
#     raise ValueError("Can't transfer to target %s" % (target,))
# ValueError: Can't transfer to target gpu

# f = function([], tensor.exp(x).transfer('cpu'))
# THEANO_FLAGS=device=cuda0 python using_gpu.py
# ERROR (theano.sandbox.gpuarray): pygpu was configured but could not be imported
# Traceback (most recent call last):
#   File "/root/Theano/theano/sandbox/gpuarray/__init__.py", line 21, in <module>
#     import pygpu
# ImportError: No module named pygpu
# [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
# Looping 1000 times took 4.698774 seconds
# Result is [ 1.54650092  1.02626526  1.73266804 ...,  2.26791716  1.88959861
#   1.06480491]
# Used the cpu

# While the time per cal appear to be much lower than the two preious
# invocations (and should indeed be lower, since we avoid a transfer) the
# massive speedup we obtained is in part dua to asynchronous nature of
# execution on GPUs, meaning that the work isn't completed yet, just
# 'launched'. We'll talk about that later.

# The object returned is a GpuArray from pygpu. It mostly acts as a numpy
# ndarray with some exceptions due to its data being on the GPU. You can
# copy it to the host and convert it to a regular ndarray by using numpy
# casting such as numpy.asarray().

# For even more speed, you can play with borrow flag. See Borrowing when
# Constructing Function Objects.

# 2.3 What Can be Accelerated on the GPU

# 2.4 GPU Async Capabilities


# 3. Software for Directly Programming a GPU


# 4. Learning to Program with PyDUDA
# 4.1 Exercise one

# 4.2 Exercise two


# 5. Note

