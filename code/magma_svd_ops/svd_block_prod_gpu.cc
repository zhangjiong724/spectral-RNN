#include <cstdio>
#include <iostream>
#include "magma_v2.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
//#define PRINT_DEBUG

using namespace tensorflow;

REGISTER_OP("SvdBlockProdGpu")
  .Input("hidden_state: float")
  .Input("householder_matrix: float")
  .Attr("is_forward: bool = true")
  .Output("output_state: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


// TODO: move the two declars to a .hpp
struct workspace {
    magmaFloat_ptr *T_array;
    magmaFloat_ptr *Tau_array;
    magmaFloat_ptr *Twork_array;
    magmaFloat_ptr *V_array;
    magmaFloat_ptr T;
    magmaFloat_ptr tau;
    magmaFloat_ptr twork;
    magmaFloat_ptr dwork;
    magmaFloat_ptr dworkvt;
};

int SvdBlockProdGpuKernelLauncher(const float* H, const float* U, float* H_out, const int n_h, const int batch, const int n_r, magma_queue_t queue, workspace ws, const bool isForward=true);

class SvdBlockProdGpuOp : public OpKernel {
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
	int _batchCount;
public:
  explicit SvdBlockProdGpuOp(OpKernelConstruction* context) : OpKernel(context) {
  	// Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_forward", &_isForward));
    // printf("Calling magma init...\n");
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

  ~SvdBlockProdGpuOp() override {
    // printf("Calling magma finalize...\n");
    if (!_queue_created) {
      // printf("destroying magma queue!\n");
      magma_queue_destroy(_queue);
      _queue_created = false;
    }
    magma_finalize();
  }

  void Compute(OpKernelContext* context) override {
	// Check number of inputs
	OP_REQUIRES(context, context->num_inputs() == 2,
                errors::InvalidArgument("SvdBlockProd expects 2 inputes."));

	// Grab the input tensor
    const Tensor& H = context->input(0);
    const Tensor& U = context->input(1);
	// Shapes of input
    const TensorShape& H_shape = H.shape();
    const TensorShape& U_shape = U.shape();

	const int n_h = H_shape.dim_size(1);
	const int n_r = U_shape.dim_size(0);
	const int batch = H_shape.dim_size(0);
	// Perform dimension check
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(H_shape),
                errors::InvalidArgument("SvdBlockProd expects H to be a 2-D matrix."));
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(U_shape),
                errors::InvalidArgument("SvdBlockProd expects U to be a 2-D matrix."));
	OP_REQUIRES(context, H_shape.dim_size(1) == U_shape.dim_size(1),
                errors::InvalidArgument("The second dimension of H and U does not match!"));

	// Create an output tensor
    Tensor* H_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, H_shape,&H_out));

	// obtain data
	const float* H_data = H.flat<float>().data();
	const float* U_data = U.flat<float>().data();
	float* H_out_data = H_out->flat<float>().data();
	/*
	// test
	int idx  =0;
	std::printf( "Before:\n");
	for(int i=0; i < H_shape.dim_size(0); i++){
		for(int j=0; j < H_shape.dim_size(1); j++){
			idx = i * H_shape.dim_size(1) + j;
			std::printf ("H(%d,%d)=%4.4f,  %4.4f\n", i, j, H.flat<float>()(idx), H_out_data[idx]);
		}
	}
	*/
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
	  _persistent_tensor_created = true;
	}
	Tensor *T = _T.AccessTensor(context);
	Tensor *tau = _tau.AccessTensor(context);
	Tensor *twork = _twork.AccessTensor(context);
	OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({ldworkvt*n_r}), &dwork));
	OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({ldworkvt*n_r}), &dworkvt));
	workspace ws;
	ws.T_array = reinterpret_cast<magmaFloat_ptr*>(T_array->flat<long long>().data());
	ws.Tau_array = reinterpret_cast<magmaFloat_ptr*>(Tau_array->flat<long long>().data());
	ws.Twork_array = reinterpret_cast<magmaFloat_ptr*>(Twork_array->flat<long long>().data());
	ws.V_array = reinterpret_cast<magmaFloat_ptr*>(V_array->flat<long long>().data());
	ws.T = T->flat<float>().data();
	ws.tau = tau->flat<float>().data();
	ws.twork = twork->flat<float>().data();
	ws.dwork = dwork.flat<float>().data();
	ws.dworkvt = dworkvt.flat<float>().data();
#if GOOGLE_CUDA
	int op_status;
	if (!_queue_created) {
	  _queue_created = true;
	  magma_queue_create(0, &_queue);
	  // printf("created magma queue at %p!\n", reinterpret_cast<void *>(_queue));
	}
    op_status = SvdBlockProdGpuKernelLauncher(H_data, U_data, H_out_data, n_h, batch, n_r, _queue, ws, _isForward);
#endif
  }
};

REGISTER_KERNEL_BUILDER(Name("SvdBlockProdGpu").Device(DEVICE_GPU), SvdBlockProdGpuOp);
