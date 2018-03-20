#BLAS2 operator:
	
    Requires:
		CUDA, cuBLAS, cudnn, tensorflow-gpu

	Compile:
		cd ./cuda_svd_ops
		make

#Running:
    python main.py test.json
