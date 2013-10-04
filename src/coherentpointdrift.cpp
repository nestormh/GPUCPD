/*
 *  Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#include "coherentpointdrift.h"
#include "decomposelu.h"
#include <pcl/common/io.h>

#include <stdio.h>
#include <boost/graph/graph_concepts.hpp>

#include <omp.h>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

CoherentPointDrift::CoherentPointDrift()
{
    // Default parameters
    m_tolerance = 1e-5f;
    m_beta = 2.0f;    
    m_maxIt = 150;
    m_omega = 0.1f;
    m_lambda = 3.0f;
    
    checkCublasReturn(cublasCreate_v2(&m_cublasHandle), __LINE__);
    checkCudaError(cudaGetDeviceProperties(&m_deviceProp, 0), __LINE__);
    
    cout << m_deviceProp.name << endl;
}

void CoherentPointDrift::allocateMatrices()
{
    // TODO: Improve the efficiency doing an allocation that takes into account the size of the block
    cudaError_t error;
    
    // Sizes of matrices
    uint32_t size_X = m_N * m_D;
    uint32_t size_Y = m_M * m_D;
    uint32_t size_W = m_M * m_D;
    uint32_t size_G = m_M * m_M;
    uint32_t size_P = m_M * m_N;
    uint32_t size_P1 = m_M;
    uint32_t size_Pt1 = m_N;
    uint32_t size_PX = m_M * m_D;
    uint32_t size_Pivot = m_M;
    uint32_t size_GW = m_M * m_D;
    uint32_t size_DxD = m_D * m_D;
    uint32_t size_MxD = m_M * m_D;
    uint32_t size_MxN = m_M * m_N;
    uint32_t size_MxM = m_M * m_M;
    uint32_t size_Dx1 = m_D;
    uint32_t size_1xD = m_D;
    uint32_t size_1xM = m_M;
    uint32_t size_1xN = m_N;
    
    // Mem sizes
    uint32_t mem_size_X = sizeof(float) * size_X;
    uint32_t mem_size_Y = sizeof(float) * size_Y;
    uint32_t mem_size_W = sizeof(float) * size_W;
    uint32_t mem_size_G = sizeof(float) * size_G;
    uint32_t mem_size_P = sizeof(float) * size_P;
    uint32_t mem_size_P1 = sizeof(float) * size_P1;
    uint32_t mem_size_Pt1 = sizeof(float) * size_Pt1;
    uint32_t mem_size_PX = sizeof(float) * size_PX;
    uint32_t mem_size_Pivot = sizeof(int) * size_Pivot;
    uint32_t mem_size_GW = sizeof(float) * size_GW;
    uint32_t mem_size_DxD = sizeof(float) * size_DxD;
    uint32_t mem_size_MxD = sizeof(float) * size_MxD;
    uint32_t mem_size_MxN = sizeof(float) * size_MxN;
    uint32_t mem_size_MxM = sizeof(float) * size_MxM;
    uint32_t mem_size_Dx1 = sizeof(float) * size_Dx1;
    uint32_t mem_size_1xD = sizeof(float) * size_1xD;
    uint32_t mem_size_1xM = sizeof(float) * size_1xM;
    uint32_t mem_size_1xN = sizeof(float) * size_1xN;
    
    // Host allocation
    h_X = (float *)malloc(mem_size_X);
    h_Y = (float *)malloc(mem_size_Y);
    float * h_W = (float *)malloc(mem_size_W);
    int * h_pivotArray = (int *)malloc(mem_size_Pivot);
    float * h_ones1xM = (float *)malloc(mem_size_1xM);
    float * h_ones1xN = (float *)malloc(mem_size_1xN);
    
    // Initialization
    for (uint32_t i = 0; i < m_N; i++) {
        h_X[IDX2C(i, 0, m_N)] = m_cloudX->at(i).x;
        h_X[IDX2C(i, 1, m_N)] = m_cloudX->at(i).y;
        h_X[IDX2C(i, 2, m_N)] = m_cloudX->at(i).z;
        h_ones1xN[i] = 1.0;
    }
    
    for (uint32_t i = 0; i < m_M; i++) {
        h_Y[IDX2C(i, 0, m_M)] = m_cloudY->at(i).x;
        h_Y[IDX2C(i, 1, m_M)] = m_cloudY->at(i).y;
        h_Y[IDX2C(i, 2, m_M)] = m_cloudY->at(i).z;
        h_ones1xM[i] = 1.0;
        h_pivotArray[i] = 1;
    }
    
    for (uint32_t i = 0; i < size_Y; i++) {
        h_W[i] = 0.0f;
    }
    
    // allocate device memory
    checkCudaError(cudaMalloc((void **) &d_X, mem_size_X), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_Y, mem_size_Y), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_W, mem_size_W), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_G, mem_size_G), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_P, mem_size_P), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_P1, mem_size_P1), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_Pt1, mem_size_Pt1), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_PX, mem_size_PX), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_pivotArray, mem_size_Pivot), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_GW, mem_size_GW), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmpDxD, mem_size_DxD), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmpMxD, mem_size_MxD), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmpMxDb, mem_size_MxD), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmpMxN, mem_size_MxN), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmpMxM, mem_size_MxM), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmpMxMb, mem_size_MxM), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmpDx1, mem_size_Dx1), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmp1xD, mem_size_1xD), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_tmp1xN, mem_size_1xN), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_ones1xM, mem_size_1xM), __LINE__);
    checkCudaError(cudaMalloc((void **) &d_ones1xN, mem_size_1xN), __LINE__);
    
    // copy host memory to device
    checkCudaError(cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice), __LINE__);
    checkCudaError(cudaMemcpy(d_Y, h_Y, mem_size_Y, cudaMemcpyHostToDevice), __LINE__);
    checkCudaError(cudaMemcpy(d_W, h_W, mem_size_W, cudaMemcpyHostToDevice), __LINE__);
    checkCudaError(cudaMemcpy(d_pivotArray, h_pivotArray, mem_size_Pivot, cudaMemcpyHostToDevice), __LINE__);
    checkCudaError(cudaMemcpy(d_ones1xM, h_ones1xM, mem_size_1xM, cudaMemcpyHostToDevice), __LINE__);
    checkCudaError(cudaMemcpy(d_ones1xN, h_ones1xN, mem_size_1xN, cudaMemcpyHostToDevice), __LINE__);
}

bool CoherentPointDrift::compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudX, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudY)
{
    // FIXME: At this point, the number of dimensions is limited to 3
    m_D = 3;
    m_N = /*5; //*/cloudX->size();
    m_M = /*6; //*/cloudY->size();
    
    m_cloudX = cloudX;
    m_cloudY = cloudY;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloudY, *cloudT);
    
    allocateMatrices();
    
    m_sigma2 = getSigma2();
    float sigma2Init = m_sigma2;
    cout << "sigma2 " << m_sigma2 << endl;
    
    getG();
    
    uint32_t iter = 0;
    float nTolerance = m_tolerance + 10.0f;
    float L = 1.0;
    float Lold;
    
    while ((iter < m_maxIt) && (nTolerance > m_tolerance) && (m_sigma2 > 1e-8)) {
        Lold = L;
        
        getP();
        
        getW();
        
        // TODO: By the moment, we do not use FGT
        exit(0);
    }
    
}

float CoherentPointDrift::getSigma2()
{
    cublasStatus_t retCublas;
    
    float sigma2 = 0.0;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    
    // Let's obtain M * trace(X' * X)
    {
        checkCublasReturn(cublasSgemm_v2(m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        m_D, m_D, m_N, &alpha, d_X, m_N, d_X, m_N, &beta, d_tmpDxD, m_D), __LINE__);
        sigma2 = m_M * trace(d_tmpDxD, m_D, m_D);
    }
    
    // Let's obtain N * trace(Y' * Y)
    {
        checkCublasReturn(cublasSgemm_v2(m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                         m_D, m_D, m_M, &alpha, d_Y, m_M, d_Y, m_M, &beta, d_tmpDxD, m_D), __LINE__);
        sigma2 += m_N * trace(d_tmpDxD, m_D, m_D);
    }
    
    // Let's obtain 2 * sum(X) * sum(Y)'
    {
        // Let's obtain sum(X)'
        float * d_sumXt = d_tmp1xD;
        checkCublasReturn(cublasSgemv_v2(m_cublasHandle, CUBLAS_OP_T,
                                         m_N, m_D, &alpha, d_X, m_N, d_ones1xN, 1, &beta, d_sumXt, 1), __LINE__);
    
        // Let's obtain sum(Y)'
        float * d_sumYt = d_tmpDx1;
        checkCublasReturn(cublasSgemv_v2(m_cublasHandle, CUBLAS_OP_T,
                                         m_M, m_D, &alpha, d_Y, m_M, d_ones1xM, 1, &beta, d_sumYt, 1), __LINE__);
        
        // Let's obtain 2 * sum(X) * sum(Y)'
        float * h_sumX, * h_sumY;
        getHostMatrix(d_sumXt, h_sumX, 1, m_D);
        getHostMatrix(d_sumYt, h_sumY, m_D, 1);
        
        float result = 0.0f;
        for (uint32_t i = 0; i < 3; i++) {
            result += h_sumX[i] * h_sumY[i];
        }
        
        sigma2 -= 2 * result;
    }
    sigma2 /= m_M * m_N * m_D;
        
    return sigma2;
}

void CoherentPointDrift::getG()
{
    launchGetG(d_G, d_Y, m_M, m_D, m_beta, m_deviceProp);
}

void CoherentPointDrift::getP()
{
    // Let's obtain G * W 
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        checkCublasReturn(cublasSgemm_v2(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        m_M, m_D, m_M, &alpha, d_G, m_M, d_W, m_M, &beta, d_GW, m_M), __LINE__);
    }
    // Let's obtain Y + G * W 
    {
        const float alpha = 1.0f;
        const float beta  = 1.0f;
        checkCublasReturn(cublasSgeam(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m_M, m_D, &alpha, d_Y, m_M, &beta, d_GW, m_M, d_tmpMxD, m_M), __LINE__);
    }
    
    launchGetP(d_X, d_tmpMxD, d_ones1xM, d_ones1xN, d_tmpMxN, d_tmp1xN, d_P, 
                m_M, m_N, m_D, m_omega, m_sigma2, m_deviceProp, m_cublasHandle);
    
    // Let's obtain P1, Pt1
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        checkCublasReturn(cublasSgemv_v2(m_cublasHandle, CUBLAS_OP_N,
                                         m_M, m_N, &alpha, d_P, m_M, d_ones1xN, 1, &beta, d_P1, 1), __LINE__);
        checkCublasReturn(cublasSgemv_v2(m_cublasHandle, CUBLAS_OP_T,
                                         m_M, m_N, &alpha, d_P, m_M, d_ones1xM, 1, &beta, d_Pt1, 1), __LINE__);
        checkCublasReturn(cublasSgemm_v2(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m_M, m_D, m_N, &alpha, d_P, m_M, d_X, m_N, &beta, d_PX, m_M), __LINE__);
    }
}

void CoherentPointDrift::getW()
{
    // (dP1 * G + lambda * sigma2 * I(MxM)) * W = (PX - dP1 * Y)
    
    // Let's obtain (dP1 * G)'
    float * d_dP1Gt = d_tmpMxM;
    checkCublasReturn(cublasSdgmm(m_cublasHandle, CUBLAS_SIDE_LEFT,
                                  m_M, m_M, d_G, m_M, d_P1, 1, d_dP1Gt, m_M), __LINE__);
    
    // Let's obtain dP1 * G + lambda * sigma2 * I(MxM)
    float * d_term1 = d_dP1Gt;
    launchAddScalarDiagonal(d_term1, m_lambda * m_sigma2, m_M, m_deviceProp);

    // Let's obtain dP * Y
    float * d_dPY = d_tmpMxD;
    checkCublasReturn(cublasSdgmm(m_cublasHandle, CUBLAS_SIDE_LEFT,
                                  m_M, m_D, d_Y, m_M, d_P1, 1, d_dPY, m_M), __LINE__);
    
    // Let's obtain PX - dP * Y
    float * d_term2 = d_tmpMxDb;
    {
        const float alpha = 1.0f;
        const float beta  = -1.0f;
        checkCublasReturn(cublasSgeam(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m_M, m_D, &alpha, d_PX, m_M, &beta, d_dPY, m_M, d_term2, m_M), __LINE__);
    }
    
    
    cout << "term1 = " << endl;
    showDeviceMatrix(d_term1, m_M, m_M);
    cout << "term2 = " << endl;
    showDeviceMatrix(d_term2, m_M, m_D);
    cout << "pivotArray = " << endl;
    showDeviceMatrix(d_pivotArray, m_M, 1);
    
    // Now: the tricky part!!! W = term1 \ term2
//     solveWinCPU(d_term1, d_term2);
//     launchQRDecomposition(d_term1, m_M, m_M, m_deviceProp);
//     cout << "LU = " << endl;
//     showDeviceMatrix(d_term1, m_M, m_M);
//     cudaInvertMatrix(m_M, d_term1);
//     
//     float * h_term1;
//     getHostMatrix(d_term1, h_term1, m_M, m_M);
//     
//     for (uint32_t i = 0; i < min(10, (int)m_M); i++) {
//         for (uint32_t j = 0; j < min(10, (int)m_M); j++) {
// //             cout << h_term1[i * m_M + j] << " ";
// //             printf("%3.4f\t", min(fabs(h_term1[j * m_M + i]), 100.0f));
// //             printf("%3.4f\t", h_term1[j * m_M + i]);
//             if (h_term1[j * m_M + i] != 0.0f)
//                 cout << "X ";
//             else cout << "0 ";
//         }
//         cout << endl;
//     }
    
    //TODO: Try with the code at 
    // https://devtalk.nvidia.com/default/topic/405197/cuda-programming-and-performance/lu-qr-and-cholesky-factorizations-using-gpu/
    
//     float * h_term2 = (float*)malloc(m_M * m_M * sizeof(float));
//     for (uint32_t i = 0; i < m_M; i++) {
//         for (uint32_t j = 0; j < m_M; j++) {
//             h_term2[i * m_M + j] = h_term1[j * m_M + i];
//         }
//     }
//     checkCudaError(cudaMemcpy(d_term1, h_term2, m_M * m_M * sizeof(float), cudaMemcpyHostToDevice), __LINE__);
    
    
//     showDeviceMatrix(d_term1, m_M, m_M);
//     int info;
//     cout << info << endl;
//     checkCublasReturn(cublasSgetrfBatched(m_cublasHandle, m_M, &d_term1, m_M, d_pivotArray, &info, 0), __LINE__);
//     cout << info << endl;
//     cout << "LU = " << endl;
//     showDeviceMatrix(d_pivotArray, m_M, 1);
    
}

template <typename T>
void CoherentPointDrift::getHostMatrix(const T* d_matrix, T * &h_matrix, const uint32_t& rows, const uint32_t& cols)
{
    cudaError_t error;
    h_matrix = new T[rows * cols];
    /*checkCudaError(*/cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(T), cudaMemcpyDeviceToHost)/*, __LINE__)*/;
}

float CoherentPointDrift::trace(const float* d_matrix, const uint32_t& rows, const uint32_t& cols)
{
    assert(rows == cols);
    
    float * h_matrix;
    getHostMatrix(d_matrix, h_matrix, rows, cols);
    
    float trace = 0.0;
    for (uint32_t i = 0; i < rows; i++) {
        trace += h_matrix[IDX2C(i, i, rows)];
    }
    
    return trace;
}

template <typename T>
void CoherentPointDrift::showDeviceMatrix(const T * d_matrix, const uint32_t & rows, const uint32_t & cols)
{
    T * h_matrix;
    getHostMatrix(d_matrix, h_matrix, rows, cols);

    
    for (uint32_t i = 0; i < min(10, (int)rows); i++) {
        for (uint32_t j = 0; j < min(10, (int)cols); j++) {
            cout << h_matrix[IDX2C(i, j, rows)] << " ";
        }
        cout << endl;
    }
}

void CoherentPointDrift::solveWinCPU(float* & d_term1, float* & d_term2)
{
    double startWallTime = omp_get_wtime();
    
    cout << __LINE__ << endl;

    cv::Mat term1(m_M, m_M, CV_32FC1);
    cv::Mat term2(m_M, m_D, CV_32FC1);
    
    cout << __LINE__ << endl;
    
    float * h_term1 = new float[m_M * m_M];
    float * h_term2 = new float[m_M * m_D];
    
    cout << __LINE__ << endl;
    
    checkCudaError(cudaMemcpy(h_term1, d_term1, m_M * m_M * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
    checkCudaError(cudaMemcpy(h_term2, d_term2, m_M * m_D * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);

    cout << __LINE__ << endl;
    
    for (uint32_t i = 0; i < m_M; i++) {
        for (uint32_t j = 0; j < m_M; j++) {
            term1.at<float>(i, j) = h_term1[j * m_M + i];
            if (j < m_D)
                term2.at<float>(i, j) = h_term2[j * m_M + i];
        }
    }
    
    cout << __LINE__ << endl;
    
    cv::Mat W;
    cv::solve(term1, term2, W, cv::DECOMP_SVD);

    cout << __LINE__ << endl;
    
    cout << "Time for " << __FUNCTION__ << ": " << omp_get_wtime() - startWallTime << endl;
    startWallTime = omp_get_wtime();
    cout << "********************************" << endl;
    
    
    cout << "W" << endl;
    cout << W << endl;
//     cout << "term1" << endl;
//     cout << term1 << endl;
//     cout << "term2" << endl;
//     cout << term2 << endl;
}