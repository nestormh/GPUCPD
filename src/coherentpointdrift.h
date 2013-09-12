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


#ifndef COHERENTPOINTDRIFT_H
#define COHERENTPOINTDRIFT_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cublas_v2.h>

extern "C" {
    void launchGetG(float * d_G, const float * d_Y, const int & M, const int & D, 
                    const float & beta, const cudaDeviceProp & deviceProp);
    void launchGetP(const float * d_X, const float * d_tmpMxD, const float * d_ones1xM, const float * d_ones1xN, float * d_tmpMxN, float * d_tmp1xN, float * d_P,
                    const int & M, const int & N, const int & D, const float & omega, const float & sigma2,
                    const cudaDeviceProp & deviceProp, const cublasHandle_t & cublasHandle);
    void checkCudaError(const cudaError_t& cudaError, const int & line);
    void checkCublasReturn(const cublasStatus_t& retCublas, const int & line);
}

class CoherentPointDrift
{
public:
    CoherentPointDrift();
    bool compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloudX, const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloudY);
    
protected:
    void allocateMatrices();
    float getSigma2();
    void getG();
    void getP();
    float trace(const float * d_matrix, const uint32_t & rows, const uint32_t & cols);
    void showDeviceMatrix(const float * d_matrix, const uint32_t & rows, const uint32_t & cols);
    void getHostMatrix(const float * d_matrix, float * &h_matrix, const uint32_t & rows, const uint32_t & cols);
    
    uint32_t m_M, m_N, m_D;
    
    // Parameters
    float m_tolerance;
    float m_beta;
    uint32_t m_maxIt;
    float m_omega;
    
    float m_sigma2;
    
    float * h_X, * h_Y, * h_W;
    float * d_X, * d_Y, * d_W, * d_G, *d_P, * d_GW, * d_P1, * d_Pt1, * d_PX;
    
    // Temporal matrices
    float * d_tmpDxD, * d_tmpMxD, * d_tmp1xD, * d_tmp1xN, * d_tmpDx1, * d_tmpMxN, * d_ones1xM, * d_ones1xN;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloudX, m_cloudY;
    
    cublasHandle_t m_cublasHandle;
    cudaDeviceProp m_deviceProp;
};

#endif // COHERENTPOINTDRIFT_H
