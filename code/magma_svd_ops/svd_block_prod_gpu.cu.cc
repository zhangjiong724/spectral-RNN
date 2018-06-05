#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_internal.h"
#include "batched_kernel_param.h"
#define THREAD_SIZE 512
#define  max_shared_bsiz 32

#define RFT_MAG_GEM
#define use_gemm_larft

extern __shared__ float shared_data[];

__global__ void ZeroTriu(float* U, const int n_h, const int n_r) {
	int col = blockIdx.x;
	for(int row = threadIdx.x; row < col; row += blockDim.x){
		U[col*n_h + row] = 0;
	}
}

__global__ void UpperTri(float* T, const int n_r, const int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N and idx%n_r > idx/n_r  ){
		T[idx] = 0;
	}
}

__global__ void ConstSet(float* tau, const float a, const int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N)	tau[idx] = a;
}

__global__ void ConstDevide(float* tau, const float a, const int N) {
	int idx = blockIdx.x;
	if(idx < N)	tau[idx] = a /  tau[idx];
}

__global__ void CalculateTau(float *tau, float* V, const int n_r, const int n_h, float init) {
	//int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.x;
	__shared__ float sdata[THREAD_SIZE];
	assert(blockDim.x == THREAD_SIZE);
	//===========================
	// init tau to be zero
	// ==========================
	//if(threadIdx.x==0){tau[col] = init;}
	//===========================
	// reduce col square
	//===========================
	// compute local col square
	float temp = 0.0;
	for(int row=threadIdx.x; row < n_h; row += blockDim.x){
		temp += V[ col*n_h + row] * V[col*n_h + row];
	}
	sdata[threadIdx.x] = temp;
	__syncthreads();
	// reduction within block (across all threads)
	int i = blockDim.x/2;
	while (i != 0){
		if (threadIdx.x < i)
			sdata[threadIdx.x] += sdata[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}
	//=========================
	// compute tau
	// =======================
	if(threadIdx.x == 0)
		tau[col] = 2.0 / sdata[0];
}


__global__ void SetAddress(float** array, float* one_matrix) {
	int idx = blockIdx.x;
	array[idx] = one_matrix;
}

void printDeviceMatrix(const float* A, int col, int row, magma_queue_t queue){
	float* hA;
	magma_smalloc_cpu(&hA, col*row);

	magma_sgetmatrix(  row, col, A, row, hA, row, queue); // copy d_a -> r
	for(int i = 0; i<row; i++){
		for(int j=0; j<col; j++){
			printf("%7.3f ", hA[j*row + i]);
		}
		printf("\n");
	}
	free(hA);
}

static __device__ void
my_slarft_recstrmv_sm32x32_device(
    int m, int n, float *tau,
    float *Trec, int ldtrec, float *Ttri, int ldttri)
{
    int tx = threadIdx.x;
    float *sdata = (float*)shared_data;
    float res;

    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate
    // one element of the column of T then move to the next column

    // read T into shared
    for (int s=0; s < n; s++)
    {
        sdata[tx + s*m] = Trec[tx + s * ldtrec];
    }
    __syncthreads();

    // perform sequence of n-1 gemv
    for (int i=0; i < n; i++)
    {
        res = MAGMA_S_ZERO;
        for (int j=0; j < i; j++)
        {
            res += sdata[tx + j * m] * Ttri[j+ i * ldttri];
        }
        __syncthreads();   // a enlever
        sdata[tx + i * m] = -tau[i] * (sdata[tx + i * m] + res);
        __syncthreads();
    }

    // write back the updated block of k column of T  multiplying by -tau
    for (int s=0; s < n; s++)
    {
        Trec[tx + s * ldtrec] = sdata[tx + s*m];
    }
}

/******************************************************************************/
__global__ void
my_slarft_recstrmv_sm32x32_kernel_batched(
    int m, int n, float **tau_array,
    float **Trec_array, int ldtrec, float **Ttri_array, int ldttri)
{
    int batchId = blockIdx.z;
    my_slarft_recstrmv_sm32x32_device(m, n, tau_array[batchId], Trec_array[batchId], ldtrec, Ttri_array[batchId], ldttri);
}


/******************************************************************************/
extern "C"
void my_magmablas_slarft_recstrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    float **tau_array,
    float **Trec_array, magma_int_t ldtrec,
    float **Ttri_array, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(1, 1, batchCount);
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(float)*(m*n);
    my_slarft_recstrmv_sm32x32_kernel_batched
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        (m, n,  tau_array, Trec_array, ldtrec, Ttri_array, ldttri);
}

/******************************************************************************/
extern "C" magma_int_t
my_magma_slarft_batched(magma_int_t n, magma_int_t k, magma_int_t stair_T,
                float **v_array, magma_int_t ldv,
                float **tau_array, float **T_array, magma_int_t ldt,
                float **work_array, magma_int_t lwork,
                magma_int_t batchCount, magma_queue_t queue)
{
    float c_one  = MAGMA_S_ONE;
    float c_zero = MAGMA_S_ZERO;

    if ( k <= 0) return 0;
    if ( stair_T > 0 && k <= stair_T) return 0;

    magma_int_t maxnb = max_shared_bsiz;

    magma_int_t info = 0;
    if (stair_T > 0 && stair_T > maxnb) {
        info = -3;
    }
    else if (lwork < k*ldt) {
        info = -10;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magma_int_t DEBUG=0;
    magma_int_t nb = stair_T == 0 ? min(k,maxnb) : stair_T;

    magma_int_t i, j, prev_n, mycol, rows;

    float **dW1_displ  = NULL;
    float **dW2_displ  = NULL;
    float **dW3_displ  = NULL;
    float **dTstep_array  = NULL;

    magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dTstep_array,  batchCount * sizeof(*dTstep_array));

    //float *Tstep =  k > nb ? work : T;
    if (k > nb)
    {
        magma_sdisplace_pointers(dTstep_array, work_array, lwork, 0, 0, batchCount, queue);
    }
    else
    {
        magma_sdisplace_pointers(dTstep_array, T_array, ldt, 0, 0, batchCount, queue);
    }

    //magma_int_t ldtstep = k > nb ? k : ldt;
    magma_int_t ldtstep = ldt; //a enlever
    // stair_T = 0 meaning all T
    // stair_T > 0 meaning the triangular portion of T has been computed.
    //                    the value of stair_T is the nb of these triangulars


    //GEMV compute the whole triangular upper portion of T (phase 1)
    // TODO addcublas to check perf

    magma_sgemm_batched( MagmaConjTrans, MagmaNoTrans,
                         k, k, n,
                         c_one,  v_array, ldv,
                                 v_array, ldv,
                         c_zero, dTstep_array, ldtstep,
                         batchCount, queue );

    magmablas_slaset_batched( MagmaLower, k, k, MAGMA_S_ZERO, MAGMA_S_ZERO, dTstep_array, ldtstep, batchCount, queue );
    // no need for it as T is expected to be lower zero
    //if (k > nb) magmablas_slaset_batched( MagmaLower, k, k, MAGMA_S_ZERO, MAGMA_S_ZERO, dTstep_array, ldtstep, batchCount, queue );


    //TRMV
    //T(1:i-1,i) := T(1:i-1,1:i-1) * W(1:i-1) i=[1:k]
    // TRMV is split over block of column of size nb
    // the update should be done from top to bottom so:
    // 1- a gemm using the previous computed columns
    //    of T to update rectangular upper protion above
    //    the triangle of my columns
    // 2- the columns need to be updated by a serial
    //    loop over of gemv over itself. since we limit the
    //    shared memory to nb, this nb column
    //    are split vertically by chunk of nb rows

    dim3 grid(1, 1, batchCount);

    for (j=0; j < k; j += nb)
    {
        prev_n =  j;
        mycol  =  min(nb, k-j);
        // note that myrow = prev_n + mycol;
        if (prev_n > 0 && mycol > 0) {
            if (DEBUG == 3) {
                printf("doing gemm on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) prev_n, (long long) mycol, (long long) 0, (long long) j );
            }

            magma_sdisplace_pointers(dW1_displ, dTstep_array, ldtstep, 0, j, batchCount, queue);
            magma_sdisplace_pointers(dW2_displ, T_array,     ldt, 0, j, batchCount, queue);
            magma_sgemm_batched( MagmaNoTrans, MagmaNoTrans,
                                 prev_n, mycol, prev_n,
                                 c_one,  T_array, ldt,
                                         dW1_displ, ldtstep,
                                 c_zero, dW2_displ, ldt,
                                 batchCount, queue );

            // update my rectangular portion (prev_n,mycol) using sequence of gemv
            magma_sdisplace_pointers(dW1_displ, dTstep_array, ldtstep, j, j, batchCount, queue);
            magma_sdisplace_pointers(dW3_displ, tau_array,  1, j, 0, batchCount, queue);

            for (i=0; i < prev_n; i += nb)
            {
                rows = min(nb,prev_n-i);
                if (DEBUG == 3) {
                    printf("        doing recstrmv on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                            (long long) rows, (long long) mycol, (long long) i, (long long) j );
                }

                if (rows > 0 && mycol > 0)
                {
                    magma_sdisplace_pointers(dW2_displ, T_array,     ldt, i, j, batchCount, queue);
                    my_magmablas_slarft_recstrmv_sm32x32_batched(rows, mycol, dW3_displ, dW2_displ, ldt, dW1_displ, ldtstep, batchCount, queue);
                }
            }
        }

        // the upper rectangular protion is updated, now if needed update the triangular portion
        if (stair_T == 0) {
            if (DEBUG == 3) {
                printf("doing strmv on the triangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) mycol, (long long) mycol, (long long) j, (long long) j );
            }

            if (mycol > 0)
            {
                magma_sdisplace_pointers(dW1_displ, dTstep_array, ldtstep, j, j, batchCount, queue);
                magma_sdisplace_pointers(dW3_displ, tau_array,  1, j, 0, batchCount, queue);
                magma_sdisplace_pointers(dW2_displ, T_array,     ldt, j, j, batchCount, queue);
                magmablas_slarft_strmv_sm32x32_batched(mycol, mycol, dW3_displ, dW1_displ, ldtstep, dW2_displ, ldt, batchCount, queue);
            }
        }
    }// end of j

    magma_free(dW1_displ);
    magma_free(dW2_displ);
    magma_free(dW3_displ);
    magma_free(dTstep_array);

    return 0;
}

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

int SvdBlockProdGpuKernelLauncher(const float* H, const float* U, float* H_out, const int n_h, const int batch, const int n_r, magma_queue_t queue, workspace ws, const bool isForward) {
	/*
	 * Computes the Hprod(U, H) = Q * H or Q^T * H, according to isForward.
	 * where Q = House(u_1)*House(u_2)*...*House(u_{n_r})
	 *
	 * H 		: the hidden states, shape = (n_h, batch)
	 * U 		: the elementary reflectors, lower triabgular, shape = (n_h ,n_r)
	 * n_h 		: hidden dimension
	 * batch 	: batch size
	 * n_r 	 	: number of Householder reflectors
	 * isForward: if doing transpose on Q
	 *
	 * According to MAGMA documents, the diagonal elements of U are supposed to be 1, but does not influence the results ??
	 * TODO : avoid the memcpy
	 */

	magma_int_t stat;
	int batchCount = 1;
	cudaDeviceSynchronize();

        // printf("using magma queue at %p!\n", reinterpret_cast<void*>(queue));

	// stat = magma_init();
	// TODO: avoid creating the queue here
	// magma_queue_t queue;
	// magma_queue_create(0, &queue);
	// if( stat != MAGMA_SUCCESS){ printf("Error init magma!\n"); return EXIT_FAILURE;}


	magmaFloat_ptr *T_array, *Tau_array, *Twork_array, *V_array;
#if 0
    magma_malloc((void**)&T_array,  batchCount * sizeof(*T_array));
    magma_malloc((void**)&Tau_array,  batchCount * sizeof(*Tau_array));
    magma_malloc((void**)&Twork_array,  batchCount * sizeof(*Twork_array));
    magma_malloc((void**)&V_array,  batchCount * sizeof(*V_array));
#else
    T_array = ws.T_array;
    Tau_array = ws.Tau_array;
    Twork_array = ws.Twork_array;
    V_array = ws.V_array;
#endif 
#if 0
	// construct alpha and fill alpha with 2.0 (assume u_i are of unit norm)
	magmaFloat_ptr T, tau;
	/*
	if( magma_smalloc(&T, n_r*n_r) != MAGMA_SUCCESS){
		printf("Error allocating T!\n");
		return EXIT_FAILURE;
	}
	*/
	T = ws.T;
	/*
	if( magma_smalloc(&tau, n_r) != MAGMA_SUCCESS){
		printf("Error allocating tau!\n");
		return EXIT_FAILURE;
	}
	*/
	tau = ws.tau;

	// allocate workspace
	magmaFloat_ptr twork, dwork, dworkvt;
	magma_int_t ldwork = n_h, ldworkvt = max(n_h,batch);
	/*
	if( magma_smalloc(&twork, n_r*n_r) != MAGMA_SUCCESS){
		printf("Error allocating twork!\n");
		return EXIT_FAILURE;
	}
	*/
	twork = ws.twork;
	/*
	if( magma_smalloc(&dwork, ldwork*n_r) != MAGMA_SUCCESS){
		printf("Error allocating dwork!\n");
		return EXIT_FAILURE;
	}
	*/
	dwork = ws.dwork;
	if( magma_smalloc(&dworkvt, ldworkvt*n_r) != MAGMA_SUCCESS){
		printf("Error allocating dworkvt!\n");
		return EXIT_FAILURE;
	}
	// dworkvt = ws.dworkvt;

#else
	magma_int_t ldwork = n_h, ldworkvt = max(n_h,batch);
	magmaFloat_ptr T = ws.T;
	magmaFloat_ptr tau = ws.tau;
	magmaFloat_ptr twork = ws.twork;
	magmaFloat_ptr dwork = ws.dwork;
	magmaFloat_ptr dworkvt = ws.dworkvt;
	/*
	magmaFloat_ptr dwork;
	if( magma_smalloc(&dwork, ldwork*n_r) != MAGMA_SUCCESS){
		printf("Error allocating dwork!\n");
		return EXIT_FAILURE;
	}
	*/

#endif
	// copy H to H_out
	magmablas_slacpy(MagmaFull, n_h, batch, H, n_h, H_out, n_h, queue);
	// compute T = inv( striu(U'U) + 0.5 * diag(U'U))
	magmablas_slacpy(MagmaFull, n_h, n_r, U, n_h, dwork, n_h, queue);

	// calculate tau[i] = 2.0/ dot(V[i],V[i])
	CalculateTau<<< n_r, THREAD_SIZE>>>(tau, dwork, n_r, n_h, 0.0); // tau = [u_i*u_i]
	//ConstSet<<< n_r, 1>>>(tau, 2.0, n_r); // tau = [u_i*u_i]
	ConstSet<<< n_r, n_r>>>(T, 0, n_r*n_r); // set T to zero

	SetAddress<<<batchCount,1>>>(T_array, T);
	SetAddress<<<batchCount,1>>>(Tau_array, tau);
	SetAddress<<<batchCount,1>>>(V_array, dwork);
	SetAddress<<<batchCount,1>>>(Twork_array, twork);

	stat = my_magma_slarft_batched(	n_h,
							n_r,
							0, 			// stair_T not sure what it does
                			V_array, n_h, 	//
                			Tau_array, 		//
							T_array, n_r, 	//
                			Twork_array, n_r*n_r,
                			1,  		// batchCount
							queue);


	// compute H = Q * H
	magma_trans_t isTrans = MagmaTrans;
	if(not isForward){
		isTrans = MagmaNoTrans;
	}

	stat |= magma_slarfb_gpu_gemm(		MagmaLeft, 		// side
								isTrans, 		// transpose
								MagmaForward, 	// Q = H(u_{n_r}) . . . H(u_2) H(u_1) (Backward)
								MagmaColumnwise,// elementary reflectors are stored
								n_h,			// number of rows of H
								batch,			// number of columns of H
								n_r,			// number of Householder reflectors
								U,				// U = (u_1, u_2,..., u_{n_r})
								n_h,			// The leading dimension of U
								T,				// block Householder T
								n_r,			// The leading dimension of T
								H_out,			// H.shape = (n_h, batch)
								n_h,			// leading dimension of H
								dwork, 			// workspace
								ldwork,			// leading dimension of workspace
								dworkvt,		// workspace 2
								ldworkvt,		// leading dimension of workspace2
								queue
								);

	// wait for all kernels in the queue
	magma_queue_sync(queue);
	cudaDeviceSynchronize();

	// free memory
#if 0
	magma_free(T_array);
	magma_free(Tau_array);
	magma_free(Twork_array);
	magma_free(V_array);

	magma_free(T);
	magma_free(tau);
	magma_free(twork);
	magma_free(dwork);
	magma_free(dworkvt);
#endif
	assert(stat == MAGMA_SUCCESS);
	// magma_queue_destroy(queue);
	// magma_finalize();

	return EXIT_SUCCESS;
}

