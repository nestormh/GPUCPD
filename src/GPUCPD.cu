/*
    Copyright 2013 Néstor Morales Hernández <email>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <stdexcept>

#include <cublas_v2.h>

#define PI 3.14159265358979

using namespace std;

extern "C"
void checkCudaError(const cudaError_t& cudaError, const int & line)
{
    if (cudaError != cudaSuccess) {
        char errorStr[1024];
        sprintf(errorStr, "Error %s, at line %s:%d", cudaGetErrorString(cudaError), __FILE__, line);
        throw (std::runtime_error(errorStr));
    }
}

extern "C"
void checkCublasReturn(const cublasStatus_t& retCublas, const int & line)
{
    if (retCublas != CUBLAS_STATUS_SUCCESS) {
        char statStr[1024];
        sprintf(statStr, "Error in cublas calculation with code %d, at line %s:%d", retCublas, __FILE__, line);
        throw std::runtime_error(statStr);
    }
}

__global__
void sumRows(const float * A, const uint32_t & M, const uint32_t & N, float * S) {
    const unsigned int idxM = (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int idxN = (blockIdx.y * blockDim.y + threadIdx.y);
    const unsigned int idx = idxN * M + idxM;
    
    if ((idxM >= M) || (idxN >= N))
        return;
}

extern "C"
void launchSumRows(const float * A, const uint32_t & M, const uint32_t & N, float * S) {
    
}

__global__
void getG(float * d_G, const float * d_Y, const unsigned int M, const unsigned int D, 
          const float gamma) {
            
    const unsigned int idxI = (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int idxJ = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if ((idxI > idxJ) || (idxI >= M) || (idxJ >= M))
        return;
    
    const unsigned int idxG = idxI * M + idxJ;
    const unsigned int idxGinv = idxJ * M + idxI;

    float value = 0.0;
    float term;
    for (int i = 0; i < D; i++) {
        term = d_Y[i * M + idxI] - d_Y[i * M + idxJ];
        value += term * term;
    }
    value = exp(value * gamma);

    d_G[idxG] = value;
    d_G[idxGinv] = value;
}

__global__
void getNumerator(const float * d_X, const float * d_YGW, float * d_numerator, const unsigned int M, const unsigned int N, 
                  const unsigned int D, const float gamma) {
    
    const unsigned int idxM = (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int idxN = (blockIdx.y * blockDim.y + threadIdx.y);
    const unsigned int idx = idxN * M + idxM;
    
    if ((idxM >= M) || (idxN >= N))
        return;
    
    float value = 0.0;
    float term;
    for (int i = 0; i < D; i++) {
        term = d_X[i * N + idxN] - d_YGW[i * M + idxM];
        value += term * term;
    }
    value *= gamma;
    
    d_numerator[idx] = exp(value);
}

__global__
void getP(const float * d_numerator, const float * d_denominator, float * d_P, const unsigned int M, const unsigned int N, const float omegaTmp) {
    
    const unsigned int idxM = (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int idxN = (blockIdx.y * blockDim.y + threadIdx.y);
    const unsigned int idx = idxN * M + idxM;
    
    if ((idxM >= M) || (idxN >= N))
        return;

    if ((idxM == 1) && (idxN == 1))  {
        printf("P[%d,%d = %d] = %f / %f = %f\n", idxM, idxN, idx, d_numerator[idx], d_denominator[idxN], d_numerator[idx] / d_denominator[idxN]);
    }
    d_P[idx] = d_numerator[idx] / (d_denominator[idxN] + omegaTmp);
}
    
extern "C"
void launchGetG(float * d_G, const float * d_Y, const int & M, const int & D, 
                const float & beta, const cudaDeviceProp & deviceProp) {
    
    const float gamma = -1 / (2 * beta * beta);
    
    int blockDimension = sqrt(deviceProp.maxThreadsPerBlock);
    int gridDimension = M / blockDimension + 1;
    
    const dim3 blockSize(blockDimension, blockDimension, 1);
    const dim3 gridSize(gridDimension, gridDimension, 1);
    
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    getG <<<gridSize, blockSize>>> (d_G, d_Y, M, D, gamma);
    cudaDeviceSynchronize(); cudaGetLastError();
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for getting G matrix = " << elapsed << endl;
    
}


extern "C"
void launchGetP(const float * d_X, const float * d_tmpMxD, const float * d_ones1xM, const float * d_ones1xN, float * d_tmpMxN, float * d_tmp1xN, float * d_P,
                const int & M, const int & N, const int & D, const float & omega, const float & sigma2,
                const cudaDeviceProp & deviceProp, const cublasHandle_t & cublasHandle) {
             
    const float gamma = -1 / (2 * sigma2);
    
    cout << "sigma2 = " << sigma2 << endl;
    cout << "gamma = " << gamma << endl;
    
    int blockDimension = sqrt(deviceProp.maxThreadsPerBlock);
    int gridDimensionX = N / blockDimension + 1;
    int gridDimensionY = M / blockDimension + 1;
    
    const dim3 blockSize(blockDimension, blockDimension, 1);
    const dim3 gridSize(gridDimensionX, gridDimensionY, 1);
    
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // First, we obtain the matrix for the numerator
    float * d_numerator = d_tmpMxN;
    const float * d_YGW = d_tmpMxD;
    getNumerator <<<gridSize, blockSize>>> (d_X, d_YGW, d_numerator, M, N, D, gamma);
    cudaDeviceSynchronize(); cudaGetLastError();

    // Now, we obtain the matrix for the denominator
    float * d_denominator = d_tmp1xN;
    {
        const float alpha = 1.0;
        const float beta = 0.0;
        checkCublasReturn(cublasSgemv_v2(cublasHandle, CUBLAS_OP_T,
                                         M, N, &alpha, d_numerator, M, d_ones1xM, 1, &beta, d_denominator, 1), __LINE__);
        //FIXME Sum this in the GetP function!!!
//         const float term2denom = (omega / (1 - omega)) * ((pow(2 * PI * sigma2, D * 0.5) * M) / N);
//         checkCublasReturn(cublasSaxpy_v2(cublasHandle, N, &term2denom, d_ones1xN, 1, d_denominator, 1), __LINE__);
    }
    
    float omegaTmp = (omega * pow((2 * PI * sigma2), (0.5 * D)) * M) / ((1 - omega) * N);
    
    // Finally, we divide the numerator by the denominator
    {
        const dim3 blockSize(1, deviceProp.maxThreadsPerBlock, 1);

        int gridDimensionX = N;
        int gridDimensionY = M / blockSize.y + 1;
        
        const dim3 gridSize(gridDimensionX, gridDimensionY, 1);
        
        cout << "omegaTmp = " << omegaTmp << endl;
        
        getP <<<gridSize, blockSize>>> (d_numerator, d_denominator, d_P, M, N, omegaTmp);
        cudaDeviceSynchronize(); cudaGetLastError();
    }
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for getting P matrix = " << elapsed << endl;
}