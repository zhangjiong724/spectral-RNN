#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

using namespace tensorflow;

REGISTER_OP("GradSvdInvProdGpu")
  .Input("hidden_state: float")
  .Input("householder_matrix: float")
  .Input("gradient_backprop: float")
  .Output("grad_hidden_state: float")
  .Output("grad_householder_matrix: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    return Status::OK();
  });

#include "tensorflow/core/framework/op_kernel.h"

int GradSvdInvProdGpuKernelLauncher(const float* H, const float* U, const float* G, float* H_grad, float* G_grad, const int n_h, const int batch, const int n_r);

class GradSvdInvProdGpuOp : public OpKernel {
public:
  explicit GradSvdInvProdGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
	// Check number of inputs
	OP_REQUIRES(context, context->num_inputs() == 3,
                errors::InvalidArgument("GradSvdInvProd expects 3 inputes."));

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
                errors::InvalidArgument("SvdInvProd expects H to be a 2-D matrix."));
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(U_shape),
                errors::InvalidArgument("SvdInvProd expects U to be a 2-D matrix."));
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(G_shape),
                errors::InvalidArgument("SvdInvProd expects G to be a 2-D matrix."));
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
#if GOOGLE_CUDA
    GradSvdInvProdGpuKernelLauncher(H_data, U_data, G_data, Grad_H_data, Grad_U_data, n_h, batch, n_r);
#endif
  }
};

REGISTER_KERNEL_BUILDER(Name("GradSvdInvProdGpu").Device(DEVICE_GPU), GradSvdInvProdGpuOp);
