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

#include "utils.h"

#include <fstream>

using namespace std;

void loadData(const std::string & dataFile, pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) 
{
    
    ifstream fin(dataFile.c_str(), ios::in);
    
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    while (! fin.eof()) {
        pcl::PointXYZ p;
        fin >> p.x;
        fin >> p.y;
        fin >> p.z;
        
        cloud->push_back(p);
    }
    
    fin.close();
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud1, 
                                                             pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud2) 
{
    // Transforms from XYZ to XYZRGB
    const uint8_t colorRed[] = { 255, 0, 0};
    const uint8_t colorBlue[] = { 0, 0, 255};
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB1 = xyz2rgbPointCloud(cloud1, colorRed);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB2 = xyz2rgbPointCloud(cloud2, colorBlue);
        
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(cloudRGB1);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(cloudRGB2);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloudRGB1, rgb1, "cloud1");
    viewer->addPointCloud<pcl::PointXYZRGB> (cloudRGB2, rgb2, "cloud2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud2");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz2rgbPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const uint8_t * color) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloudRGB->reserve(cloud->size());
    for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = cloud->begin(); it != cloud->end(); it++) {
        pcl::PointXYZRGB p(color[0], color[1], color[2]);
        p.x = it->x;
        p.y = it->y;
        p.z = it->z;
        
        cloudRGB->push_back(p);
    }
    
    return cloudRGB;
}