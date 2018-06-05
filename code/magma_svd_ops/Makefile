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
TF_LIB = `python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"`

# specify your magma dir and MKL dir here
MAGMA  =/work/03941/jiongdys/maverick/magma
CUDADIR =/opt/apps/cuda/8.0/
MKL = /opt/apps/intel/15/composer_xe_2015.3.187/mkl/include
MKLLIB = /opt/apps/intel/15/composer_xe_2015.3.187/mkl/lib/intel64

CC        = gcc -O2 -pthread
GPUCC     = nvcc
CFLAGS    = -O3 -std=c++11 -L$(TF_LIB) -I$(TF_INC) -I$(TF_INC)/external/nsync/public  -DADD_  -ltensorflow_framework 


MAGMA_I   = -I${MAGMA}/include -I${MAGMA}/testing -I${MAGMA}/control -I${MKL}
MAG_FLAGS =  -m64 -DNDEBUG -O3   -Wall -Wshadow -DMAGMA_NOAFFINITY -pedantic -Wno-long-long -DHAVE_CUBLAS -DMIN_CUDA_ARCH=300  -c ${MAGMA_I} -I${CUDADIR}/include
CMAG_FLAGS= -m64  -fPIC ${MAGMA_I} -Wl,-rpath,${MAGMA}/lib -L${MAGMA}/lib  -L${MKLLIB} -lstdc++ -lm   #-framework Accelerate -lblas_fix

GPUCFLAGS = -c -arch=sm_30 --expt-relaxed-constexpr ${MAGMA_I}

LFLAGS    = -pthread -shared -fPIC
GPULFLAGS = -x cu -Xcompiler -fPIC
GPUDEF    = -DGOOGLE_CUDA=1
CGPUFLAGS = -lcuda -lcublas -lmagma # -lmagmablas

SRC       = svd_block_prod_gpu.cc
GPUSRC    = svd_block_prod_gpu.cu.cc
PROD      = svd_block_prod_gpu.so
GPUPROD   = svd_block_prod_cu_gpu.o

GRAD_SRC       = grad_svd_block_prod_gpu.cc
GRAD_GPUSRC    = grad_svd_block_prod_gpu.cu.cc
GRAD_PROD      = grad_svd_block_prod_gpu.so
GRAD_GPUPROD   = grad_svd_block_prod_cu_gpu.o

default: gpu  gpu-grad

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS)  $(GPUSRC) $(GPULFLAGS) $(GPUDEF) -o $(GPUPROD)
	$(CXX) $(CFLAGS) ${CMAG_FLAGS} $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) $(OSFLAGS) $(GPUDEF) -o $(PROD)

gpu-grad:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GRAD_GPUSRC) $(GPULFLAGS) $(GPUDEF) -o $(GRAD_GPUPROD)
	$(CXX) $(CFLAGS) ${CMAG_FLAGS} $(GRAD_SRC) $(GRAD_GPUPROD) $(LFLAGS) $(CGPUFLAGS) $(OSFLAGS) $(GPUDEF) -o $(GRAD_PROD)

clean:
	rm -f $(TEST_PROD) $(TEST_FINAL) $(PROD) $(GPUPROD) $(GRAD_PROD) $(GRAD_GPUPROD)
