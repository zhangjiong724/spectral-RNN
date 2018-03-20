#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

using namespace tensorflow;

REGISTER_OP("SvdInvProdGpu")
  .Input("hidden_state: float")
  .Input("householder_matrix: float")
  .Output("output_state: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


int SvdInvProdGpuKernelLauncher(const float* H, const float* U, float* H_out, const int n_h, const int batch, const int n_r);

class SvdInvProdGpuOp : public OpKernel {
public:
  explicit SvdInvProdGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
	// Check number of inputs
	OP_REQUIRES(context, context->num_inputs() == 2,
                errors::InvalidArgument("SvdInvProd expects 2 inputes."));

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
                errors::InvalidArgument("SvdInvProd expects H to be a 2-D matrix."));
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(U_shape),
                errors::InvalidArgument("SvdInvProd expects U to be a 2-D matrix."));
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
#if GOOGLE_CUDA
	int op_status;
    op_status = SvdInvProdGpuKernelLauncher(H_data, U_data, H_out_data, n_h, batch, n_r );
#endif
  }
};

REGISTER_KERNEL_BUILDER(Name("SvdInvProdGpu").Device(DEVICE_GPU), SvdInvProdGpuOp);
