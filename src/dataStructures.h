// ============================================================================
//  
//  Project 2.1: 2D Feature Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_2D_Feature_Tracking
//
// ============================================================================

// image frame and evaluation result data structure definition

#ifndef dataStructures_h
#define dataStructures_h

#include <string>
#include <vector>
#include <opencv2/core.hpp>


struct DataFrame {  // represents the available sensor information at the same time instance
    
    // raw image data
    std::string imgFilename;  // camera image file name
    cv::Mat cameraImg;  // camera image
    
    // keypoints and keypoint descriptors
    std::vector<cv::KeyPoint> keypoints;  // 2D keypoints within camera image
    cv::Mat descriptors;  // keypoint descriptors for each keypoint within camera image
    std::vector<cv::DMatch> kptMatches;  // keypoint matches between previous and current frame
};

struct EvalResults {  // represents the evaluation results collected over a variable number of data frames

    // image name
    std::string imgFilename;  // camera image file name

    // configuration parameters
    std::string detectorType;  // keypoint destector type
    bool bFocusOnVehicle;  // ROI filter to let only keypoints within a given target bounding box pass
    bool bLimitKpts;  // force limitation of detected keypoints => only for debugging => should be false
    std::string descExtractorType;  // keypoint descriptor extractor type
    std::string matcherType;  // descriptor matcher type
    std::string descriptorType;  // descriptor type
    std::string selectorType;  // selector type

    // evaluation results
    int numKeypoints;  // number of keypoints found in the image
    int numKeypointsInROI;  // number of detected keypoints within the region of interest
    int numKeypointsInROILimited;  // limited number of detected keypoints within the region of interest
    int numDescMatches;  // number of matched keypoints within the region of interest
    double meanDetectorResponse;  // mean keypont detector response
    double meanKeypointDiam; // mean keypoint diameter
    double varianceKeypointDiam; // variance of keypoint diameters
    double t_detKeypoints;  // processing time needed for keypoint detection (all keypoints)
    double t_descKeypoints;  // processing time needed for keypoint descriptor extraction (keypoints in ROI)
    double t_matchDescriptors;  // processing time needed for keypoint descriptor matching (keypoints in ROI)
    double t_sum_det_desc;  // t_detKeypoints + t_descKeypoints
    double t_sum_det_desc_match;  // t_detKeypoints + t_descKeypoints + t_matchDescriptors
};

#endif  /* dataStructures_h */
