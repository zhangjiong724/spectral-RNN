# Makefile
OS := $(shell uname)

ifeq ($(OS),Darwin)  # Mac OS X
	OSFLAGS = -undefined dynamic_lookup
	CXX = clang++
else
	CXX = g++
	OSFLAGS = 
endif

TF_INC = `python -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB= `python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'`

CC        = gcc -O2 -pthread
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -L$(TF_LIB) -I$(TF_INC)/external/nsync/public  -ltensorflow_framework 
GPUCFLAGS = -c --expt-relaxed-constexpr
LFLAGS    = -pthread -shared -fPIC
GPULFLAGS = -x cu -Xcompiler -fPIC
GPUDEF    = -DGOOGLE_CUDA=1
CGPUFLAGS = -lcuda -lcublas

SRC       = svd_prod_gpu.cc
GPUSRC    = svd_prod_gpu.cu.cc
PROD      = svd_prod_gpu.so
GPUPROD   = svd_prod_cu_gpu.o

GRAD_SRC       = grad_svd_prod_gpu.cc
GRAD_GPUSRC    = grad_svd_prod_gpu.cu.cc
GRAD_PROD      = grad_svd_prod_gpu.so
GRAD_GPUPROD   = grad_svd_prod_cu_gpu.o

INV_SRC       = svd_inv_prod_gpu.cc
INV_GPUSRC    = svd_inv_prod_gpu.cu.cc
INV_PROD      = svd_inv_prod_gpu.so
INV_GPUPROD   = svd_inv_prod_cu_gpu.o

INV_GRAD_SRC       = grad_svd_inv_prod_gpu.cc
INV_GRAD_GPUSRC    = grad_svd_inv_prod_gpu.cu.cc
INV_GRAD_PROD      = grad_svd_inv_prod_gpu.so
INV_GRAD_GPUPROD   = grad_svd_inv_prod_cu_gpu.o

default: gpu  gpu-grad inv-gpu inv-gpu-grad

cpu:
	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) $(OSFLAGS) -o $(PROD)

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) $(GPUDEF) -o $(GPUPROD)
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) $(OSFLAGS) $(GPUDEF) -o $(PROD)

gpu-grad:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GRAD_GPUSRC) $(GPULFLAGS) $(GPUDEF) -o $(GRAD_GPUPROD)
	$(CXX) $(CFLAGS)  $(GRAD_SRC) $(GRAD_GPUPROD) $(LFLAGS) $(CGPUFLAGS) $(OSFLAGS) $(GPUDEF) -o $(GRAD_PROD)

inv-gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(INV_GPUSRC) $(GPULFLAGS) $(GPUDEF) -o $(INV_GPUPROD)
	$(CXX) $(CFLAGS)  $(INV_SRC) $(INV_GPUPROD) $(LFLAGS) $(CGPUFLAGS) $(OSFLAGS) $(GPUDEF) -o $(INV_PROD)

inv-gpu-grad:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(INV_GRAD_GPUSRC) $(GPULFLAGS) $(GPUDEF) -o $(INV_GRAD_GPUPROD)
	$(CXX) $(CFLAGS)  $(INV_GRAD_SRC) $(INV_GRAD_GPUPROD) $(LFLAGS) $(CGPUFLAGS) $(OSFLAGS) $(GPUDEF) -o $(INV_GRAD_PROD)

clean:
	rm -f $(PROD) $(GPUPROD) $(GRAD_PROD) $(GRAD_GPUPROD) $(INV_PROD) $(INV_GPUPROD) $(INV_GRAD_PROD) $(INV_GRAD_GPUPROD)
