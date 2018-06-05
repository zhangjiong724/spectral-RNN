#include <cstdio>
#include <iostream>
#include "magma_v2.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

using namespace tensorflow;

REGISTER_OP("GradSvdBlockProdGpu")
  .Input("hidden_state: float")
  .Input("householder_matrix: float")
  .Input("gradient_backprop: float")
  .Attr("is_forward: bool = true")
  .Output("grad_hidden_state: float")
  .Output("grad_householder_matrix: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    return Status::OK();
  });

#include "tensorflow/core/framework/op_kernel.h"

struct grad_workspace {
    magmaFloat_ptr *T_array;
    magmaFloat_ptr *Tau_array;
    magmaFloat_ptr *Twork_array;
    magmaFloat_ptr *V_array;
    magmaFloat_ptr T;
    magmaFloat_ptr tau;
    magmaFloat_ptr twork;
    magmaFloat_ptr dwork;
    magmaFloat_ptr dworkvt;
    magmaFloat_ptr Q_grad;
    magmaFloat_ptr UT;
};

int GradSvdBlockProdGpuKernelLauncher(const float* H, const float* U, const float* G, float* H_grad, float* G_grad, const int n_h, const int batch, const int n_r, magma_queue_t queue, grad_workspace ws, const bool isForward);

class GradSvdBlockProdGpuOp : public OpKernel {
private:
	bool _isForward;
	magma_queue_t _queue;
	bool _queue_created;
	bool _persistent_tensor_created;
	PersistentTensor _T_array;
	PersistentTensor _Tau_array;
	PersistentTensor _Twork_array;
	PersistentTensor _V_array;
	PersistentTensor _T;
	PersistentTensor _tau;
	PersistentTensor _twork;
	PersistentTensor _Q_grad;
	PersistentTensor _UT;
	int _batchCount;
public:
  explicit GradSvdBlockProdGpuOp(OpKernelConstruction* context) : OpKernel(context) {
  	// Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_forward", &_isForward));
    // printf("Calling magma init in grad...\n");
    magma_int_t stat = magma_init();
    if( stat != MAGMA_SUCCESS){ printf("Error init magma!\n"); }
    _queue_created = false;
    _persistent_tensor_created = false;
    // create array space
    _batchCount = 1;
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT64, TensorShape({_batchCount}), &_T_array, nullptr));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT64, TensorShape({_batchCount}), &_Tau_array, nullptr));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT64, TensorShape({_batchCount}), &_Twork_array, nullptr));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT64, TensorShape({_batchCount}), &_V_array, nullptr));
  }

  ~GradSvdBlockProdGpuOp() override {
    // printf("Calling magma finalize in grad...\n");
    if (!_queue_created) {
      // printf("destroying magma queue!\n");
      magma_queue_destroy(_queue);
      _queue_created = false;
    }
    magma_finalize();
  }

  void Compute(OpKernelContext* context) override {
	// Check number of inputs
	OP_REQUIRES(context, context->num_inputs() == 3,
                errors::InvalidArgument("GradSvdBlockProd expects 3 inputes."));

	// Grab the input tensor
    const Tensor& H = context->input(0);
    const Tensor& U = context->input(1);
    const Tensor& G = context->input(2);
    auto input = H.flat<float>();

	// Shapes of input
    const TensorShape& H_shape = H.shape();
    const TensorShape& U_shape = U.shape();
    const TensorShape& G_shape = G.shape();

	const int n_h = H_shape.dim_size(1);
	const int n_r = U_shape.dim_size(0);
	const int batch = H_shape.dim_size(0);
	// Perform dimension check
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(H_shape),
                errors::InvalidArgument("SvdBlockProd expects H to be a 2-D matrix."));
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(U_shape),
                errors::InvalidArgument("SvdBlockProd expects U to be a 2-D matrix."));
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(G_shape),
                errors::InvalidArgument("SvdBlockProd expects G to be a 2-D matrix."));
	OP_REQUIRES(context, H_shape.dim_size(1) == U_shape.dim_size(1),
                errors::InvalidArgument("The second dimension of H and U does not match!"));
	OP_REQUIRES(context, G_shape.dim_size(0) == H_shape.dim_size(0),
                errors::InvalidArgument("The first dimension of G and H does not match!"));
	OP_REQUIRES(context, G_shape.dim_size(1) == H_shape.dim_size(1),
                errors::InvalidArgument("The second dimension of G and H does not match!"));

	// Create an output tensor
    Tensor* Grad_H = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, H_shape,&Grad_H));
    Tensor* Grad_U = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, U_shape,&Grad_U));

	// obtain data
	const float* H_data = H.flat<float>().data();
	const float* U_data = U.flat<float>().data();
	const float* G_data = G.flat<float>().data();
	float* Grad_H_data = Grad_H->flat<float>().data();
	float* Grad_U_data = Grad_U->flat<float>().data();
	// Allocate temp tensors
	Tensor *T_array = _T_array.AccessTensor(context);
	Tensor *Tau_array = _Tau_array.AccessTensor(context);
	Tensor *Twork_array = _Twork_array.AccessTensor(context);
	Tensor *V_array = _V_array.AccessTensor(context);
	Tensor dwork;
	Tensor dworkvt;
	int ldwork = n_h, ldworkvt = std::max(n_h,batch);
	if (!_persistent_tensor_created) {
	  OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, TensorShape({n_r*n_r}), &_T, nullptr));
	  OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, TensorShape({n_r}), &_tau, nullptr));
	  OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, TensorShape({n_r*n_r}), &_twork, nullptr));
	  OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, TensorShape({n_h*n_h}), &_Q_grad, nullptr));
	  OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, TensorShape({n_h*n_r}), &_UT, nullptr));
	  _persistent_tensor_created = true;
	}
	Tensor *T = _T.AccessTensor(context);
	Tensor *tau = _tau.AccessTensor(context);
	Tensor *twork = _twork.AccessTensor(context);
	Tensor *Q_grad = _Q_grad.AccessTensor(context);
	Tensor *UT = _UT.AccessTensor(context);
	OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({ldworkvt*n_r}), &dwork));
	OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({ldworkvt*n_r}), &dworkvt));
	grad_workspace ws;
	ws.T_array = reinterpret_cast<magmaFloat_ptr*>(T_array->flat<long long>().data());
	ws.Tau_array = reinterpret_cast<magmaFloat_ptr*>(Tau_array->flat<long long>().data());
	ws.Twork_array = reinterpret_cast<magmaFloat_ptr*>(Twork_array->flat<long long>().data());
	ws.V_array = reinterpret_cast<magmaFloat_ptr*>(V_array->flat<long long>().data());
	ws.T = T->flat<float>().data();
	ws.tau = tau->flat<float>().data();
	ws.twork = twork->flat<float>().data();
	ws.dwork = dwork.flat<float>().data();
	ws.dworkvt = dworkvt.flat<float>().data();
	ws.Q_grad = Q_grad->flat<float>().data();
	ws.UT = UT->flat<float>().data();
#if GOOGLE_CUDA
	if (!_queue_created) {
	  _queue_created = true;
	  magma_queue_create(0, &_queue);
	  // printf("created magma queue at %p!\n", reinterpret_cast<void *>(_queue));
	}
    GradSvdBlockProdGpuKernelLauncher(H_data, U_data, G_data, Grad_H_data, Grad_U_data, n_h, batch, n_r, _queue, ws, _isForward);
#endif
  }
};

REGISTER_KERNEL_BUILDER(Name("GradSvdBlockProdGpu").Device(DEVICE_GPU), GradSvdBlockProdGpuOp);
