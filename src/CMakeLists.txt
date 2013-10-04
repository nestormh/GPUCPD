set(CCFILES
    main.cpp
    utils.cpp
    coherentpointdrift.cpp
    decomposelu.cpp
  
#     glapack/gpu_lapack/gpu_lapack_internal.cpp
#     glapack/gpu_lapack/gpu_sgeqrf.cpp
#     glapack/gpu_lapack/gpu_sgetrf.cpp
#     glapack/gpu_lapack/gpu_spotrf.cpp

    LU-Decomposition/LUDecomposition/deviceQuery.cpp
#     LU-Decomposition/LUDecomposition/main.cpp
    LU-Decomposition/LUDecomposition/matrix_func.cpp
#     LU-Decomposition/LUDecomposition/mem.cpp
    LU-Decomposition/LUDecomposition/util.cpp


)

set (HFILES
)

set (CUHFILES

)

set(CUFILES
    GPUCPD.cu
    
#     LU-Decomposition/LUDecomposition/call_kernel.cu
#     LU-Decomposition/LUDecomposition/kernel.cu
    LU-Decomposition/LUDecomposition/lud_block.cu
#     LU-Decomposition/LUDecomposition/lud_block_kernel.cu
#     LU-Decomposition/LUDecomposition/nvidia_kernel.cu


#     glapack/gpu_lapack/gpu_batch_sswap.cu
#     glapack/gpu_lapack/gpu_blas3.cu
#     glapack/gpu_lapack/gpu_misc.cu
#     glapack/gpu_lapack/gpu_transpose.cu

)

set(HFILES_PUBLIC
)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)

set(CUDAGDB_OPTS -g -G)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_executable(testGPUCPD
  ${CUFILES} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
    OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS}
#   OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS} ${CUDA_BIND_COMPILER}
#   ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS} ${CUDAGDB_OPTS}
)
  
include_directories(testGPUCPD
#     GPULibSVM
#     ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
#     emon.a
    ${CUDA_INCLUDE_DIRS}
    cuda_common
    glapack
    LU-Decomposition/LUDecomposition
    LU-Decomposition
)

# set(CUDA_CURAND_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand.so)

target_link_libraries(testGPUCPD
#   ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${BOOST_LIBRARIES}
#   ${CUDA_CURAND_LIBRARIES}
)
CUDA_ADD_CUBLAS_TO_TARGET(testGPUCPD)
