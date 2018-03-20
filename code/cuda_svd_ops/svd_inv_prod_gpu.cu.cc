#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


int inline Hprod(cublasHandle_t handle, float* H, const float* u, float* alpha, const int k, const int n_h, const int batch) {

	cublasStatus_t stat;
	float aa = 0;
	float bb = 0;
	float cc = -1.0;
	// aa = 2.0 / u^T * u
	stat = cublasSdot (handle, k, u + n_h - k, 1, u + n_h - k, 1, &aa);
	aa = 2.0 / aa;
	// make sure that leading (n_h-k) entrees of u are 0
	//stat = cublasSscal(handle, n_h-k, &bb, u, 1);
	// compute alpha = aa * H^T * u
	stat = cublasSgemv(handle, CUBLAS_OP_T, n_h, batch,
						&aa, H, n_h,
						u, 1,
						&bb, alpha, 1);
	// update H
	stat = cublasSger(handle, n_h, batch, &cc, u, 1, alpha, 1, H, n_h);

	if (stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	return EXIT_SUCCESS;
}

// host function for CUDA kernels
int SvdInvProdGpuKernelLauncher(const float* H, const float* U, float* H_out, const int n_h, const int batch, const int n_r) {
	cublasStatus_t stat;
	cudaError_t cudaStat;
	cublasHandle_t handle;
	// creat handle
	stat = cublasCreate_v2(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed on SvdInvProd\n"); return EXIT_FAILURE; }
	// allocate alpha
	float* alpha;
	cudaStat = cudaMalloc ((void**)&alpha, batch*sizeof(float));
	if (cudaStat != cudaSuccess) { printf ("device memory allocation of alpha failed"); return EXIT_FAILURE; }
	// begin computation
	stat = cublasScopy(handle, n_h * batch , H, 1, H_out, 1); // fill H_out with H

	for(int r=n_r-1; r >= 0; r--) {
		Hprod(handle, H_out, U + n_h*r, alpha, n_h - r, n_h, batch);
	}
	cudaFree(alpha);
	cublasDestroy(handle);
	return EXIT_SUCCESS;
}

#endif
