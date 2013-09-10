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
#ifndef UTILS_GPU_CPD_H
#define UTILS_GPU_CPD_H

#include <stdint.h>
#include <vector>
#include <string.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

extern "C" {
    void loadData(const std::string & dataFile, pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud1, 
                                                                 pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz2rgbPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const uint8_t * color);
}

#endif