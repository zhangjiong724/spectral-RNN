import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf

def Hprod(H, u, k):
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    alpha = 2* np.dot(H[:, -k:], u[-k:]) / np.dot(u[-k:],u[-k:]) # alpha.shape = (batch,)
    H_out = H.copy()
    H_out[:, -k:] -= np.outer(alpha, u[-k:])
    return H_out

def tf_Hprod(H, u, k):
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    u_square = tf.tensordot(u[-k:],u[-k:],1)
    alpha = 2* tf.tensordot(H[:, -k:],  u[-k:],1) / u_square # alpha.shape = (batch,)
    H_update = tf.identity(H[:,-k:])
    #H_update = tf.subtract(H_update, tf.einsum('i,j->ij',alpha, u[-k:]))
    H_update = tf.subtract(H_update, tf.expand_dims(alpha,1) * tf.expand_dims(u[-k:],0))

    H_out = tf.concat([H[:,0 :-k], H_update], axis=1)
    return H_out

def Hgrad(H, u, G, k):  # unused
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    # G.shape = (batch, n_h)
    alpha = 2* np.dot(H[:, -k:], u[-k:]) # alpha.shape = (batch,)
    beta = 2* np.dot(G[:, -k:], u[-k:]) # beta.shape = (batch,)
    u_bar = -np.dot(alpha,G) - np.dot(beta,H) + np.dot(alpha,beta)*u  # sum of gradient within the batch: averaging needed???
    G_out = G.copy()
    G_out -=  np.outer(beta,u)
    return G_out, u_bar  # G_out.shape = (batch, n_h); u_bar.shape = (n_h,)

def tf_Hgrad(H, u, G, k):
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    # G.shape = (batch, n_h)
    u_square = tf.tensordot(u[-k:],u[-k:],1)
    alpha = 2* tf.tensordot(H[:, -k:],  u[-k:],1) / u_square # alpha.shape = (batch,)
    beta = 2* tf.tensordot(G[:, -k:],  u[-k:],1) / u_square # beta.shape = (batch,)

    u_bar = -tf.tensordot(alpha,G[:,-k:],1) - tf.tensordot(beta,H[:,-k:],1) + tf.tensordot(alpha,beta,1)*u[-k:]  # sum of gradient within the batch: averaging needed???
    u_bar = tf.concat([u[0 :-k],u_bar], axis=0)


    G_update = tf.identity(G[:,-k:])
    delta_G = tf.expand_dims(beta, 1) * tf.reshape(u[-k: ], shape=(1 , k))
    G_update = tf.subtract(G_update,  delta_G)
    G_out = tf.concat([G[:,0 :-k], G_update], axis=1)

    return G_out, u_bar  # G_out.shape = (batch, n_h); u_bar.shape = (n_h,)
###### FP definition ########

def np_svdProd(H,U):
    #U_shape = U.get_shape().as_list()
    U_shape = U.shape
    n_r = U_shape[0]; n_h = U_shape[1]
    assert( H.shape[1] == n_h)
    H_copy = H.copy()
    for i in range(0, n_r):
        H_copy = Hprod(H_copy, U[i], n_h-i)
    return H_copy

def np_svdProd_inv(H,U):
    #U_shape = U.get_shape().as_list()
    U_shape = U.shape
    n_r = U_shape[0]; n_h = U_shape[1]
    assert( H.shape[1] == n_h)
    H_copy = H.copy()
    for i in range(n_r-1,-1,-1):
        H_copy = Hprod(H_copy, U[i], n_h-i)
    return H_copy
###### BP definition #########

def svdProdGrad(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]

    #return H, grad

    U_shape = U.get_shape().as_list()
    n_r = U_shape[0]; n_h = U_shape[1]
    #batch = H.get_shape().as_list()[0]
    #assert( H.get_shape().as_list()[1] == n_h)

    H_hist = [tf.zeros_like(H, dtype=tf.float32)]*n_r

    H_hist[0] = tf.add(H_hist[0], H)
    for i in range(0, n_r-1):
        H_hist[i+1] = tf_Hprod( H_hist[i], U[i,:], n_h-i)

    U_bar = [tf.zeros_like(U[0,:], dtype=tf.float32)] * n_r
    G = grad

    for i in range(n_r-1, -1, -1):
        G, U_bar[i] = tf_Hgrad(H_hist[i], U[i], G, n_h-i)
    U_grad = tf.stack(U_bar)

    return G, U_grad #the propagated gradient with respect to the first and second argument respectively

def svdProdGrad_inv(op, grad):
    H = op.inputs[0]
    U = op.inputs[1]

    U_shape = U.get_shape().as_list()
    n_r = U_shape[0]; n_h = U_shape[1]

    H_hist = [tf.zeros_like(H, dtype=tf.float32)]*n_r

    H_hist[n_r-1] = tf.add(H_hist[n_r-1], H)
    for i in range(n_r-1, 0, -1):
        H_hist[i-1] = tf_Hprod( H_hist[i], U[i,:], n_h-i)

    U_bar = [tf.zeros_like(U[0,:], dtype=tf.float32)] * n_r
    G = grad

    for i in range(0, n_r):
        G, U_bar[i] = tf_Hgrad(H_hist[i], U[i], G, n_h-i)
    U_grad = tf.stack(U_bar)

    return G, U_grad #the propagated gradient with respect to the first and second argument respectively

###### TF operator definition #######

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_svdProd(H,U, name=None):

    with ops.name_scope(name, "svdProd",[H,U] )as name:
        z = py_func(np_svdProd,
                        [H,U],
                        [tf.float32],
                        name=name,
                        grad=svdProdGrad)  # <-- here's the call to the gradient
        return z[0]

def tf_svdProd_inv(H,U, name=None):

    with ops.name_scope(name, "svdProd_inv",[H,U] )as name:
        z = py_func(np_svdProd_inv,
                        [H,U],
                        [tf.float32],
                        name=name,
                        grad=svdProdGrad_inv)  # <-- here's the call to the gradient
        return z[0]
