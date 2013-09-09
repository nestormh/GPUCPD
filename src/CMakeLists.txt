set(CCFILES
  main.cpp
)

set (HFILES
)

set (CUHFILES
)

set(CUFILES
)

set(HFILES_PUBLIC
)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

cuda_add_executable(testGPUCPD
  ${CUFILES} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS} ${CUDA_BIND_COMPILER}
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
