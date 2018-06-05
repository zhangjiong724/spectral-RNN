import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
np.set_printoptions(threshold=np.nan)
n_h = 128; n_b = 512; n_r = 16
print_res = (n_h * n_r < 100)


def Hgrad(H, u, G, k):
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    # G.shape = (batch, n_h)
    alpha = 2* np.dot(H[:, -k:], u[-k:]) / np.dot(u[-k:],u[-k:]) # alpha.shape = (batch,)
    beta = 2* np.dot(G[:, -k:], u[-k:]) / np.dot(u[-k:],u[-k:]) # beta.shape = (batch,)
    u_bar = np.zeros_like(u)
    u_bar[-k:] += -np.dot(alpha,G[:,-k:]) - np.dot(beta,H[:,-k:]) + np.dot(alpha,beta)*u[-k:]  # sum of gradient within the batch: averaging needed???
    G_out = G.copy()
    G_out[:,-k:] -=  np.outer(beta,u[-k:])
    return G_out, u_bar  # G_out.shape = (batch, n_h); u_bar.shape = (n_h,)

rng = np.random.RandomState(13)
H_ = rng.uniform(-np.sqrt(6. / (n_b + n_h)), np.sqrt(6. / (n_b + n_h)), (n_b, n_h)).astype(np.float32)

U_full = rng.normal(0, 0.01, (n_h, n_r)).astype(np.float32)
U_ = np.tril(U_full)
norms_U_ = np.linalg.norm(U_, axis=0)
#U_ = np.transpose(1. / norms_U_ * U_)
U_ = np.transpose(U_)

T = np.triu( np.dot(U_, U_.T))
for ii in range(n_r):
    T[ii,ii] /=2

T_inverse = np.linalg.inv(T)

#for ii in range(n_r):
#    for jj in range(n_r):
#        print ' %6.3f'%(T_inverse[ii,jj]),
#    print ''



if print_res:
    print "T: ", T
    print "T_inverse: ", np.linalg.inv(T)

    print "H: ",H_
    print "U: ",U_


############################################################
#Forward
############################################################
H1 = [H_]*(n_r+1)

for i in range(0,n_r):
    alpha = np.dot(H1[i], U_[i])
    #print 'alpha: ', 2*alpha
    H1[i+1] = H1[i] - 2 * np.outer(alpha, U_[i]) / np.dot(U_[i],U_[i])

H2 = H1[-1]


G = np.ones_like(H_)
Grad_U = np.ones_like(U_)

for i in range(n_r-1, -1, -1):
    G, Grad_U[i] = Hgrad(H1[i], U_[i], G, n_h-i)

if print_res:
    print "Hprod: ",H2
    print "Grad_H: ",G
    print "Grad_U: ",Grad_U

################ BLAS3 VER ########################
#Blas_G = np.ones_like(H_.T)
#Blas_U = U_.T
#Blas_H = H_.T
#Grad_Q = np.dot( Blas_G , Blas_H.T)
#print "Grad_Q: ", Grad_Q
#print "U * T: ", np.dot(Blas_U, np.linalg.inv(T.T))

#R =np.dot( np.dot( Grad_Q.T , Blas_U ),  np.linalg.inv(T.T))
#print "R: ", R
#print "U * T^T: ", np.dot(Blas_U, np.linalg.inv(T.T).T)
#S =np.dot( np.dot( Grad_Q , Blas_U ), np.linalg.inv(T.T).T)

#M =np.dot(np.dot( np.linalg.inv(T.T) , Blas_U.T) , R)
#print "M: ", M
#i_lower = np.tril_indices(2, -1)
#M[i_lower] = M.T[i_lower]

#print "P: ", M

#Hprod_BLAS3 = np.eye(3) - np.dot(np.dot( Blas_U, np.linalg.inv(T.T)), Blas_U.T)
#Hprod_BLAS3 = np.dot( Hprod_BLAS3, Blas_H)

#Grad_U_BLAS3 = np.dot( Blas_U, M )- S - R

#print "Hprod_BLAS3: ", Hprod_BLAS3.T
#print "Grad_U_BLAS3: ", Grad_U_BLAS3.T


############################################################
#Backward
############################################################
H1 = [H_]*(n_r+1)
for i in range(n_r-1,-1,-1):
    alpha = np.dot(H1[i+1], U_[i])
    H1[i] = H1[i+1] - 2 * np.outer(alpha, U_[i]) / np.dot(U_[i],U_[i])


H2_back = H1[0]


G_back = np.ones_like(H_)
Grad_U_back = np.ones_like(U_)

for i in range(0,n_r):
    G_back, Grad_U_back[i] = Hgrad(H1[i+1], U_[i], G_back, n_h-i)

if print_res:
    print "H_inv_prod: ",H2_back
    print "Grad_inv_H: ",G_back
    print "Grad_inv_U: ",Grad_U_back
############################################################
############################################################
svd_block_prod_module = tf.load_op_library('./svd_block_prod_gpu.so')
############################################################

grad_svd_block_prod_module = tf.load_op_library('./grad_svd_block_prod_gpu.so')

@ops.RegisterGradient("SvdBlockProdGpu")
def _svd_block_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    isForward = op.get_attr("is_forward")
    return grad_svd_block_prod_module.grad_svd_block_prod_gpu(H,U,grad, isForward)
############################################################
svd_prod_module = tf.load_op_library('../cuda_svd_ops/svd_prod_gpu.so')
############################################################
grad_svd_prod_module = tf.load_op_library('../cuda_svd_ops/grad_svd_prod_gpu.so')

@ops.RegisterGradient("SvdProdGpu")
def _svd_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    return grad_svd_prod_module.grad_svd_prod_gpu(H,U,grad)
############################################################
svd_inv_prod_module = tf.load_op_library('../cuda_svd_ops/svd_inv_prod_gpu.so')
grad_svd_inv_prod_module = tf.load_op_library('../cuda_svd_ops/grad_svd_inv_prod_gpu.so')

@ops.RegisterGradient("SvdInvProdGpu")
def _svd_inv_prod_gpu_grad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]
    return grad_svd_inv_prod_module.grad_svd_inv_prod_gpu(H,U,grad)
############################################################
with tf.Session() as sess:

    H = tf.constant(H_, dtype=tf.float32)
    U = tf.constant(U_, dtype = tf.float32)
    V = tf.constant(U_, dtype = tf.float32)

    U = tf.matrix_band_part(U, 0, -1) # upper triangular
    V = tf.matrix_band_part(V, 0, -1) # upper triangular

    z = svd_block_prod_module.svd_block_prod_gpu(H,U, True)
    blas2_z = svd_prod_module.svd_prod_gpu(H,U)

    z2 = svd_block_prod_module.svd_block_prod_gpu(H,V, False)
    blas2_z2 = svd_inv_prod_module.svd_inv_prod_gpu(H,V)
    
    gr = tf.gradients(z, [H,U])
    blas2_gr = tf.gradients(blas2_z, [H,U])

    gr2 = tf.gradients(z2, [H,V])
    blas2_gr2 = tf.gradients(blas2_z2, [H,V])
    
    tf.global_variables_initializer().run()

    if print_res:
        print('H,U and product: ',H.eval(), U.eval(),z.eval())
        print('BLAS2 H,U and product: ',H.eval(), U.eval(),blas2_z.eval())
        print('grad_H, grad_U: ' ,gr[0].eval(), gr[1].eval())

        print('H,U and product: ',z.eval(), V.eval(),z2.eval())
        print('grad_H, grad_U: ' ,gr2[0].eval(), gr2[1].eval())


    print "Forward Hprod error:", np.amax( abs(H2 - z.eval()))
    print "Forward Hgrad G error:", np.amax( abs(G - gr[0].eval()))
    print "Forward Hgrad U error:", np.amax( abs(Grad_U - gr[1].eval()))
    print "BLAS2 Forward Hprod error:", np.amax( abs(H2 - blas2_z.eval()))
    print "BLAS2 Forward Hgrad G error:", np.amax( abs(G - blas2_gr[0].eval()))
    print "BLAS2 Forward Hgrad U error:", np.amax( abs(Grad_U - blas2_gr[1].eval()))
    print "Backward Hprod error:", np.amax( abs(H2_back - z2.eval()))
    print "Backward Hgrad G error:", np.amax( abs(G_back - gr2[0].eval()))
    print "Backward Hgrad U error:", np.amax( abs(Grad_U_back  - gr2[1].eval()))
    print "BLAS2 Backward Hprod error:", np.amax( abs(H2_back - blas2_z2.eval()))
    print "BLAS2 Backward Hgrad G error:", np.amax( abs(G_back - blas2_gr2[0].eval()))
    print "BLAS2 Backward Hgrad U error:", np.amax( abs(Grad_U_back - blas2_gr2[1].eval()))
