#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


int inline Hprod(cublasHandle_t handle,const float* H_in, float* H_out, const float* u, float* alpha, const int k, const int n_h, const int batch) {

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
						&aa, H_in, n_h,
						u, 1,
						&bb, alpha, 1);
	// update H
	stat = cublasScopy(handle, n_h * batch , H_in, 1, H_out, 1); // fill H_out with H_in
	stat = cublasSger(handle, n_h, batch, &cc, u, 1, alpha, 1, H_out, n_h);

	if (stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	return EXIT_SUCCESS;
}

int inline Hgrad(cublasHandle_t handle,const float* H, const float* u, float* G, float* u_grad, float* alpha, float* beta, const int k, const int n_h, const int batch) {

	cublasStatus_t stat;
	float aa = 0;
	float zero = 0;
	float neg_one = -1.0;
	float pos_one = 1.0;
	float alpha_dot_beta = 0;
	// aa = 2.0 / u^T * u
	stat = cublasSdot (handle, k, u + n_h - k, 1, u + n_h - k, 1, &aa);
	aa = 2.0 / aa;
	// make sure that leading (n_h-k) entrees of u are 0
	//stat = cublasSscal(handle, n_h-k, &bb, u, 1);
	// compute alpha = aa * H^T * u
	stat = cublasSgemv(handle, CUBLAS_OP_T, n_h, batch,
						&aa, H, n_h,
						u, 1,
						&zero, alpha, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("alpha failed\n"); return EXIT_FAILURE; }
	// compute beta = aa * G^T * u
	stat = cublasSgemv(handle, CUBLAS_OP_T, n_h, batch,
						&aa, G, n_h,
						u, 1,
						&zero, beta, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("beta failed\n"); return EXIT_FAILURE; }
	// compute dot(alpha, beta)
	stat = cublasSdot (handle, batch, alpha, 1, beta, 1, &alpha_dot_beta);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("alpha dot beta failed\n"); return EXIT_FAILURE; }
	// u_grad = - G * alpha + 0 * u_grad
	stat = cublasSgemv(handle, CUBLAS_OP_N, n_h, batch,
						&neg_one, G, n_h,
						alpha, 1,
						&zero, u_grad, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("u_grad alpha failed\n"); return EXIT_FAILURE; }
	// u_grad = - G * alpha + 1 * u_grad
	stat = cublasSgemv(handle, CUBLAS_OP_N, n_h, batch,
						&neg_one, H, n_h,
						beta, 1,
						&pos_one, u_grad, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("u_grad beta failed\n"); return EXIT_FAILURE; }
	// u_grad = alpha_dot_beta * u + 1 * u_grad
	stat = cublasSaxpy(handle, n_h, &alpha_dot_beta, u, 1, u_grad, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("u_grad u failed\n"); return EXIT_FAILURE; }
	// zero out first n_h - k entrees --- there is better way!
	stat = cublasSscal(handle, n_h - k, &zero, u_grad, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("u_grad zero out failed\n"); return EXIT_FAILURE; }
	// update G
	stat = cublasSger(handle, n_h, batch, &neg_one, u, 1, beta, 1, G, n_h);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("G update failed\n"); return EXIT_FAILURE; }
	return EXIT_SUCCESS;
}
// host function for CUDA kernels
int GradSvdProdGpuKernelLauncher(const float* H, const float* U, const float* G, float* H_grad, float* U_grad, const int n_h, const int batch, const int n_r) {
	cublasStatus_t stat;
	cudaError_t cudaStat;
	cublasHandle_t handle;
	// creat handle
	stat = cublasCreate_v2(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed on GradSvdProd\n"); return EXIT_FAILURE; }
	// allocate alpha
	float* alpha;
	cudaStat = cudaMalloc ((void**)&alpha, batch*sizeof(float));
	if (cudaStat != cudaSuccess) { printf ("device memory allocation of alpha failed"); return EXIT_FAILURE; }
	float* beta;
	cudaStat = cudaMalloc ((void**)&beta, batch*sizeof(float));
	if (cudaStat != cudaSuccess) { printf ("device memory allocation of beta failed"); return EXIT_FAILURE; }
	// allocate H_hist
	float* H_hist;
	cudaStat = cudaMalloc ((void**)&H_hist, (n_r-1)*batch*n_h*sizeof(float));
	if (cudaStat != cudaSuccess) { printf ("device memory allocation of H_hist failed"); return EXIT_FAILURE; }
	// begin computation
	Hprod(handle, H, H_hist, U, alpha, n_h, n_h, batch);
	for(int r=1; r < n_r-1; r++) {
		Hprod(handle, H_hist + (r-1)*batch*n_h, H_hist + r*batch*n_h, U + n_h*r, alpha, n_h - r, n_h, batch);
	}

	stat = cublasScopy(handle, n_h * batch , G, 1, H_grad, 1); // fill H_out with H_in

	for(int r=n_r-1; r >0; r--) {
		Hgrad(handle, H_hist + (r-1)*batch*n_h, U + n_h*r, H_grad, U_grad + n_h*r, alpha, beta, n_h - r, n_h, batch);
	}
	Hgrad(handle, H, U, H_grad, U_grad, alpha, beta, n_h, n_h, batch);


	cudaFree(alpha);
	cudaFree(beta);
	cudaFree(H_hist);
	cublasDestroy(handle);
	return EXIT_SUCCESS;
}
#endif
