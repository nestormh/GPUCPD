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

#include <iostream>
#include <stdio.h>

#include "utils.h"

#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include "coherentpointdrift.h"

// #include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;

const string X_DATA_PATH = "/home/nestor/Dropbox/projects/GPUCPD/data/bunny1.txt";
const string Y_DATA_PATH = "/home/nestor/Dropbox/projects/GPUCPD/data/bunny2.txt";

int main(int argV, char * argC[]) {
        
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudX, cloudY;
    loadData(X_DATA_PATH, cloudX);
    loadData(Y_DATA_PATH, cloudY);
    
    // NOTE: Debug only
//     {
//         cloudX->clear();
//         cloudY->clear();
//         
//         pcl::PointXYZ p;
//         p.x = 1.0f;
//         p.y = 2.0f;
//         p.z = 3.0f;
//         for (uint32_t i = 0; i < 4; i++) {
//             cloudX->push_back(p);
//             cloudY->push_back(p);
//         }
//     }
    // End of NOTE
    
    cout << "Points X " << cloudX->size() << endl;
    cout << "Points Y " << cloudY->size() << endl;
        
    CoherentPointDrift cpd;
    cpd.compute(cloudX, cloudY);
    
//     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = rgbVis(cloudX, cloudY);
//     while (!viewer->wasStopped ())
//     {
//         viewer->spinOnce (100);
//         boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//     }
    
    return 0;
}