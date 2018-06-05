import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
import os
#from svd_ops import tf_svdProd, tf_svdProd_inv
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

############################################################
############ BLAS3 version of SVD ops ######################
############################################################
svd_block_prod_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/magma_svd_ops/svd_block_prod_gpu.so')
############################################################
grad_svd_block_prod_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/magma_svd_ops/grad_svd_block_prod_gpu.so')

@ops.RegisterGradient("SvdBlockProdGpu")
def _svd_block_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    isForward = op.get_attr("is_forward")
    return grad_svd_block_prod_module.grad_svd_block_prod_gpu(H,U,grad, isForward)
############################################################
############ BLAS2 version of SVD ops ######################
############################################################
svd_prod_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/cuda_svd_ops/svd_prod_gpu.so')
grad_svd_prod_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/cuda_svd_ops/grad_svd_prod_gpu.so')
@ops.RegisterGradient("SvdProdGpu")
def _svd_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    return grad_svd_prod_module.grad_svd_prod_gpu(H,U,grad)

############################################################
svd_inv_prod_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/cuda_svd_ops/svd_inv_prod_gpu.so')
grad_svd_inv_prod_module = tf.load_op_library(os.path.dirname(os.path.abspath(__file__)) + '/cuda_svd_ops/grad_svd_inv_prod_gpu.so')
@ops.RegisterGradient("SvdInvProdGpu")
def _svd_inv_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    return grad_svd_inv_prod_module.grad_svd_inv_prod_gpu(H,U,grad)

############################################################

class SpectralRNNCell(tf.contrib.rnn.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    """
    """
    n_h: hidden state size
    n_o: output size
    n_r: reflector size
    variables: pass a dictionary of Variables, and we will not create new ones
    backend: blas3, blas2 or python
    """

    def __init__(self, n_h, n_r = None, r_margin = 0.01,
                 linear_out=False, activation=tf.nn.relu, variables=None, backend="blas3"):
        self._n_h = n_h
        self._n_r = n_r or n_h//4
        self._r_margin = r_margin

        self._linear_out = linear_out
        self._activation = activation
        self._variables = variables
        self._backend = backend

    @property
    def state_size(self):
        return self._n_h

    @property
    def reflector_size(self):
        return self._n_r

    @property
    def output_size(self):
        return self._n_h

    def __call__(self, inputs, state, scope=None):
        """
        recur*: r
        state*: mu
        stats*: phi
        _mavg_alphas: alpha vector
        """
        with tf.variable_scope(scope or type(self).__name__):
            # Compute the output.
            """
            o_t = W^o mu_t + b^o
            """
            output = _svdlinear([inputs, state], self._n_h, self._n_r, True, r=self._r_margin,  scope='output', variables=self._variables, backend=self._backend)
            #output = _linear([inputs, state], self._n_h, True, scope='output')


            if not self._linear_out:
                output = self._activation(output, name='output_act')
            """
            o_t and mu_t
            """
            return (output, output)


# No longer publicly expose function in tensorflow.
def _svdlinear(args, output_size, reflector_size, bias, bias_start=0.0, sig_mean = 1.0, r = 0.01, scope=None, variables=None, backend="blas3"):
    """Linear map with svd operator

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      sig_mean: initial and "mean" value of singular values, usually set to 1.0,
                for ResNet should be set to 0.0
      r: singular margin, the allowed margin for singular values
      scope: VariableScope for the created subgraph; defaults to "Linear".
      variables: pass a dictionary of Variables, and we will not create new ones
      backend: blas3, blas2 or python

    Returns:
      A 2D Tensor with shape [batch x output_size] 

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape or unknown backend is passed
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    dtype = [a.dtype for a in args][0]
    # computation for svd:Hprod
    with tf.variable_scope(scope or "svdHprod"):
        if variables:
            U_full = variables["Householder_U_full"]
        else:
            U_full = tf.get_variable(
                "Householder_U_full", [reflector_size, output_size], dtype=dtype)
        U = tf.matrix_band_part(U_full, 0, -1) # upper triangular
        if variables: 
            p = variables["p"]
        else:
            p = tf.get_variable(
                "p", [ output_size], dtype=dtype,
                initializer=tf.constant_initializer(np.zeros(output_size)))
        Sig = 2*r*(tf.sigmoid(p) - 0.5) + sig_mean
        if variables:
            V_full = variables["Householder_V_full"]
        else:
            V_full = tf.get_variable(
                "Householder_V_full", [reflector_size, output_size], dtype=dtype)
        V = tf.matrix_band_part(V_full, 0, -1) # upper triangular


        if backend == "python":
            svd_term = tf_svdProd( args[1], V) # python operator
            svd_term = tf.multiply(svd_term, Sig)
            svd_term = tf_svdProd_inv( svd_term, U) # python operator
        elif backend == "blas2":
            svd_term = svd_prod_module.svd_prod_gpu( args[1], V) # BLAS2 operator
            svd_term = tf.multiply(svd_term, Sig)
            svd_term = svd_inv_prod_module.svd_inv_prod_gpu( svd_term, U) # BLAS2 operator
        elif backend == "blas3":
            svd_term = svd_block_prod_module.svd_block_prod_gpu( args[1], V, True) # BLAS3 operator
            svd_term = tf.multiply(svd_term, Sig)
            svd_term = svd_block_prod_module.svd_block_prod_gpu( svd_term, U, False) # BLAS3 operator
        else:
            raise ValueError("Unknown backend " + backend)



    # Now the computation for the rest
    with tf.variable_scope(scope or "svdLinear"):
        if variables:
            matrix = variables["Matrix"]
        else:
            matrix = tf.get_variable(
                "Matrix", [args[0].shape[1].value, output_size], dtype=dtype)
        res = tf.matmul(args[0], matrix)
        if not bias:
            return res + svd_term
        if variables:
            bias_term = variables["Bias"]
        else:
            bias_term = tf.get_variable(
                "Bias", [output_size],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype)
            )
    return res + bias_term + svd_term

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" %
                str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" %
                str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(bias_start, dtype=dtype)
        )
    return res + bias_term
