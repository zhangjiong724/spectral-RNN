import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

def Hgrad(H, u, G, k):
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    # G.shape = (batch, n_h)
    alpha = 2* np.dot(H[:, -k:], u[-k:]) # alpha.shape = (batch,)
    beta = 2* np.dot(G[:, -k:], u[-k:]) # beta.shape = (batch,)
    u_bar = np.zeros_like(u)
    u_bar[-k:] += -np.dot(alpha,G[:,-k:]) - np.dot(beta,H[:,-k:]) + np.dot(alpha,beta)*u[-k:]  # sum of gradient within the batch: averaging needed???
    G_out = G.copy()
    G_out[:,-k:] -=  np.outer(beta,u[-k:])
    return G_out, u_bar  # G_out.shape = (batch, n_h); u_bar.shape = (n_h,)

n_h = 3; n_b = 2; n_r = 2
rng = np.random.RandomState(13)
H_ = rng.uniform(-np.sqrt(6. / (n_b + n_h)), np.sqrt(6. / (n_b + n_h)), (n_b, n_h)).astype(np.float32)

U_full = rng.normal(0, 0.01, (n_h, n_r)).astype(np.float32)
U_ = np.tril(U_full)
norms_U_ = np.linalg.norm(U_, axis=0)
U_ = np.transpose(1. / norms_U_ * U_)

print H_
print U_

H1 = [H_]*(n_r+1)

for i in range(0,n_r):
    alpha = np.dot(H1[i], U_[i])
    print 'alpha: ', 2*alpha
    H1[i+1] = H1[i] - 2 * np.outer(alpha, U_[i])

H2 = H1[-1]
print H2

for i in range(n_b):
    print np.dot(H2[i],H2[i]) - np.dot(H_[i], H_[i])

G = np.ones_like(H_)
Grad_U = np.ones_like(U_)

for i in range(n_r-1, -1, -1):
    G, Grad_U[i] = Hgrad(H1[i], U_[i], G, n_h-i)

print G
print Grad_U
############################################################
############################################################

grad_svd_prod_module = tf.load_op_library('./grad_svd_prod_gpu.so')

@ops.RegisterGradient("SvdProdGpu")
def _svd_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    return grad_svd_prod_module.grad_svd_prod_gpu(H,U,grad)
############################################################
svd_prod_module = tf.load_op_library('./svd_prod_gpu.so')
############################################################

grad_svd_inv_prod_module = tf.load_op_library('./grad_svd_inv_prod_gpu.so')

@ops.RegisterGradient("SvdInvProdGpu")
def _svd_inv_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    return grad_svd_inv_prod_module.grad_svd_inv_prod_gpu(H,U,grad)
############################################################
svd_inv_prod_module = tf.load_op_library('./svd_inv_prod_gpu.so')
############################################################

with tf.Session() as sess:

    H = tf.constant(H_, dtype=tf.float32)
    U = tf.constant(U_, dtype = tf.float32)

    U = tf.matrix_band_part(U, 0, -1) # upper triangular

    z = svd_prod_module.svd_prod_gpu(H,U)
    z2 = svd_inv_prod_module.svd_inv_prod_gpu(z,U)
    gr = tf.gradients(z, [H,U])
    gr2 = tf.gradients(z2, [z,U])
    tf.global_variables_initializer().run()

    print('H,U and product: ',H.eval(), U.eval(),z.eval())
    print('grad_H, grad_U: ' ,gr[0].eval(), gr[1].eval())

    print('H,U and product: ',H.eval(), U.eval(),z2.eval())
    print('grad_H, grad_U: ' ,gr2[0].eval(), gr2[1].eval())
