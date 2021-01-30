// ============================================================================
//  
//  Project 2.1: 2D Feature Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_2D_Feature_Tracking
//
// ============================================================================

// helper function declarations for 2D keypoint detection and 2D feature matching

#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/circular_buffer/base.hpp>

#include "dataStructures.h"

bool compareKeypointResponse(const cv::KeyPoint & p1, const cv::KeyPoint & p2);
double detKeypointsShiTomasi(std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, bool bVis=false);
double detKeypointsHarris(std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, bool bVis=false);
double detKeypointsModern(std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, std::string detectorType, bool bVis=false);
double descKeypoints(std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, cv::Mat & descriptors, std::string descExtractorType);
double matchDescriptors(
    std::vector<cv::KeyPoint> & kPtsSource, std::vector<cv::KeyPoint> & kPtsRef, cv::Mat & descSource, cv::Mat & descRef,
    std::vector<cv::DMatch> & matches, std::string descriptorType, std::string matcherType, std::string selectorType);
void exportResultsToCSV(const std::string fullFilename, boost::circular_buffer<EvalResults> & resultBuffer);
void exportOverallResultsToCSV(const std::string fullFilename, std::vector<boost::circular_buffer<EvalResults>> & evalResultBuffers);

#endif /* matching2D_hpp */
