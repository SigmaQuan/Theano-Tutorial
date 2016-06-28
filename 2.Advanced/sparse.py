# Sparse

# In general, sparse matrices provide the same functionality as regular
# matrices. The difference lies in the way the elements of sparse matrices
# are represented and stored in memory. Only the non-zero elements of the
# latter are stored. This has some potential advantages: first, this may
# obviously lead to reduced memory usage and, second, clever storage
# methods may lead to reduced computation time through the use of sparse
# specific algorithms. We usually refer to the generically stored matrices
# as dense matrices.

# Theano's sparse package provides efficient algorithms, but its use is not
# recommended in all cases or for all matrices. As an obvious example,
# consider the case where the sparsity proportion is very low. The sparsity
# proportion refers to the ratio of the number of zero elements to the
# number of all elements in a matrix. A low sparsity proportion may result
# in the use of more space in memory since not only the actual data is
# stored, but also the position of nearly every element of the matrix. This
# would also require more computation time whereas a dense matrix
# representation along with regular optimized algorithms might do a better
# job. Other examples may be found at the nexus of the specific purpose
# and structure of the matrices. More documentation may be found in the
# SciPy Sparse Reference.
# http://docs.scipy.org/doc/scipy/reference/sparse.html

# Since sparse matrices are not stored in contiguous arrays, there are
# several ways to represent them in memory. This is usually designated by
# so-called format of the matrix. Since Theano's sparse matrix package is
# based on the SciPy sparse package, complete information about sparse
# matrices can be found in the SciPy documentation. Like SciPy, Theano does
# not implement sparse formats for arrays with a number of dimensions
# different from two.

# So far, theano implements two formats of sparse matrix: csc and csr.
# Those are almost identical except csc is based on the columns of the
# matrix and csr is based on its rows. They both have the same purpose:
# to provide for the use of efficient algorithms performing linear algebra
# operations. A disadvantage is that they fall to give an efficient way to
# modify the sparsity structure of the underlying matrix, i.e. adding new
# elements. This means that if you are planning to add new elements in a
# sparse matrix very often in your computational graph, perhaps a tensor
# variable could be a better choice.

# More documentation may be found in the Sparse Library Reference.
# http://deeplearning.net/software/theano/library/sparse/index.html#libdoc-sparse

# Before going further, here are the import statements that are assumed
# for the rest of the tutorial:

import theano
import numpy as np
import scipy.sparse as sp
from theano import sparse


# 1. Compressed Sparse Format
# Theano supports two compressed sparse formats: csc and csr, respectively
# based on columns and rows. They have both the same attributes: data,
# indices, indptr and shape.
# (1) The data attribute is a one-dimensional ndarray which contains all
# the non-zero elements of the sparse matrix.
# (3) The indices and indptr attributes are used to store the position of
# the data in the sparse matrix.
# (3) The shape attribute is exactly the same as the shape attribute of a
# dense (i.e. generic) matrix. It can be explicitly specified at the
# creation of a sparse matrix if it cannot be infered from the first three
# attributes.

# 1.1 Which format should I use?
# At the end, the format does not affect the length of the data and indices
# attributes. They are both completly fixed by the number of elements you
# want to store. The only thing that changes with the format is indptr. In
# csc format, the matrix is compressed along columns so a lower number of
# columns will result in less memory use. On the other hand, with the csr
# format, the matrix is compressed along the rows and with a matrix that
# have a lower number of rows, csr format is better choice. So here is the
# rule:

# Note: If shape[0]>shape[1], use csc format, otherwise, usr csr.as

# Sometimes, since the sparse module is young, ops does not exist for both
# format. So here is what may be most relevent rule:

# Note: Use the format compatible with the ops in your computation graph.

# The documentation about the ops and their supported format may be found
# in the Sparse Library Reference.
# http://deeplearning.net/software/theano/library/sparse/index.html#libdoc-sparse


# 2. Handing Sparse in Theano
# Most of the ops in  Theano depend on the format of the sparse matrix.
# That is why there are two kinds of constructors of sparse variable:
# csc_matrix and csr_matrix. These can be called with the usual name
# and dtype parameters, but no broadcastable flags are allowed. This is
# forbidden since the sparse package, as the SciPy sparse module, does not
# provide any way to handle a number of dimensions different from two. The
# set of all accepted dtype for the sparse matrices can be found in
# sparse.all_dtypes.
print sparse.all_dtypes

# 2.1 To and Fro
# To move back and forth from a dense matrix to a sparse matrix
# representation, Theano provides the dense_from_sparse and csc_from_dense
# functions. No additional detail must be provided. Here is an example
# that performs a full cycle from sparse to sparse:
x = sparse.csc_matrix(name='x', dtype='float32')
y = sparse.dense_from_sparse(x)
z = sparse.csc_from_dense(y)

# 2.2 Properties and Construction
# Although sparse variables do not allow direct to their properties, this
# can be accomplished using the csm_properties function. This will return
# a tuple of one-dimensional tensor variables that represents the internal
# characteristics of the sparse matrix.

# In order to reconstruct a sparse matrix from some properties, the
# function CSC and CSR can be used. This will create the sparse matrix in
# desired format. As an example, the following code reconstructs a csc
# matrix into a csr one.
x = sparse.csc_matrix(name='x', dtype='int64')
data, indices, indptr, shape = sparse.csm_properties(x)
y = sparse.CSR(data, indices, indptr, shape)
f = theano.function([x], y)
a = sp.csc_matrix(np.asarray([[0, 1, 1], [0, 0, 0], [1, 0, 0]]))  #, dtype='int64'
print a.toarray()
print f(a).toarray()
# The last example shows that one format can be obtained from transposition
# of the other. Indeed, when calling the transpose function, the sparse
# characteristics of the resulting matrix cannot be the same as the one
# provided as input.

# 2.3 Structured Operation
# Several ops are set to make use of the very peculiar structure of the
# sparse matrices. These ops are said to be structured and simply do not
# perform any computations on the zero elements of the sparse matrix. They
# can be though as being applied only to the data attribute of the latter.
# Note that these structured ops provide a structured gradient. More
# explication below.
x = sparse.csc_matrix(name='x', dtype='float32')
y = sparse.structured_add(x, 2)
f = theano.function([x], y)
a = sp.csc_matrix(np.asarray([[0, 0, -1], [0, -2, 1], [3, 0, 0]], dtype='float32'))
print a.toarray()
print f(a).toarray()

# 2.4 Gradient
# The gradients of the ops in the sparse module can also be structured.
# Some ops provide a flag to indicate if the gradient is to be
# structured or not. The documentation can be used to determine if the
# gradient of an op is regular or strutured or if its implementation can be
# modified. Similarly to structured ops, when a structured gradient is
# calculated, the computation is done only for the non-zero elements of the
# sparse matrix.

# More documentation regarding the gradient of specific ops can be found in
# the Sparse Library Reference.
# http://deeplearning.net/software/theano/library/sparse/index.html#libdoc-sparse
