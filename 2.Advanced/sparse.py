# Sparse

# In general, sparse matrices provide the same functionality as regular
# matrices.

import theano
import numpy as np
import scipy.sparse as sp
from theano import sparse


# 1. Compressed Sparse Format

# 1.1 Which format should I use?


# 2. Handing Sparse in Theano
print sparse.all_dtypes

# 2.1 To and Fro

x = sparse.csc_matrix(name='x', dtype='float32')
y = sparse.dense_from_sparse(x)
z = sparse.csc_from_dense(y)


# 2.2 Properties and Construction

x = sparse.csc_matrix(name='x', dtype='int64')
data, indices, indptr, shape = sparse.csm_properties(x)
y = sparse.CSR(data, indices, indptr, shape)
f = theano.function([x], y)
a = sp.csc_matrix(np.asarray([[0, 1, 1], [0, 0, 0], [1, 0, 0]]))  #, dtype='int64'
print a.toarray()
print f(a).toarray()



# 2.3 Structured Operation

x = sparse.csc_matrix(name='x', dtype='float32')
y = sparse.structured_add(x, 2)
f = theano.function([x], y)
a = sp.csc_matrix(np.asarray([[0, 0, -1], [0, -2, 1], [3, 0, 0]], dtype='float32'))
print a.toarray()
print f(a).toarray()

# 2.4 Gradient


