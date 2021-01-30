// ============================================================================
//  
//  Project 2.1: 2D Feature Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_2D_Feature_Tracking
//
// ============================================================================

/* INCLUDES FOR THIS PROJECT */
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
#include "matching2D.hpp"

using namespace std;


/* GLOBAL VARIABLES */
string dataPath = "../"; // data location


/* 2D FEATURE TRACKING FUNCTION */
int fun_2DFeatureTracking(
    vector<boost::circular_buffer<EvalResults>> & evalResultBuffers,
    string detectorType = "FAST",
    bool bFocusOnVehicle = true,
    bool bLimitKpts = false,
    int maxKeypoints = 50,
    string descExtractorType = "BRIEF",
    string matcherType = "MAT_BF",
    string descriptorType = "DES_BINARY",
    string selectorType = "SEL_KNN",
    bool bVis = true,
    bool bVisDebug = false,
    bool bExportResultsToCSV = true)
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // camera data frames
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000";  // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // using boost circular data buffer class template
    // https://www.boost.org/doc/libs/1_65_1/doc/html/circular_buffer.html
    int dataBufferSize = 2; // no. of images which are held in memory (ring buffer) at the same time
    //vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time (constantly growing! => replace with ringbuffer)
    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize); // create circular buffer for DataFrame structures

    // print data buffer capacity
    cout << "Circular data buffer capacity in use = "
        << dataBuffer.size()
        << " of max. "
        << dataBuffer.capacity()
        << " data frames."
        << endl;

    // visualization (debugging)
    //bool bVis = true;        // visualize results
    //bool bVisDebug = false;  // visualize intermediate results (for debugging or to look into details)

    // circular data buffer to hold evaluation results
    //bool bExportResultsToCSV = true;
    int resultBufferSize = imgEndIndex - imgStartIndex + 1;
    boost::circular_buffer<EvalResults> resultBuffer(resultBufferSize);

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // create data frame and push it back to the tail of the (circular) data frame buffer, or ringbuffer, resp.
        DataFrame frame;
        dataBuffer.push_back(frame);  // push back empty data frame to the tail of the data frame ringbuffer

        // store filename without path and grayscale image in the last element of the data frame ringbuffer
        (dataBuffer.end() - 1)->imgFilename = imgNumber.str() + imgFileType;
        (dataBuffer.end() - 1)->cameraImg = imgGray;

        // clear temporary variables that are no longer needed (avoid memory leaks)
        img.release();
        imgGray.release();

        // create structure to hold the evaluation results and push it back to tail of the (circular) result buffer, or ringbuffer, resp.
        EvalResults results;
        resultBuffer.push_back(results);  // push back empty results structure to the tail fo the result ringbuffer

        // store filename without path in the last element of the result ringbuffer
        (resultBuffer.end() - 1)->imgFilename = imgNumber.str() + imgFileType;
        
        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        // Print data buffer capacity in use
        cout << "Circular data buffer capacity in use = "
            << dataBuffer.size()
            << " of max. "
            << dataBuffer.capacity()
            << " data frames."
            << endl;

        /* DETECT 2D KEYPOINTS IN CURRENT IMAGE */

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        // Select keypoint detector type
        //string detectorType = "SIFT";  // options: "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "KAZE", "AKAZE", "SIFT", "SURF"
        
        // store selected detector type in the last element of result ringbuffer
        (resultBuffer.end() - 1)->detectorType = detectorType;
        cout << "Seletect keypoint detector tpye = " << detectorType << endl;

        // create empty feature list for current image
        vector<cv::KeyPoint> keypoints;

        // Initialize processing time for keypoint detection
        double t_detKeypoints = 0.0;
        
        // detect keypoints
        if (detectorType.compare("SHITOMASI") == 0)
        {
            try
            {
                // Detect keypoints using Shi-Tomasi detector
                t_detKeypoints = detKeypointsShiTomasi(keypoints, (dataBuffer.end() - 1)->cameraImg, bVisDebug);
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            try
            {
                // detect keypoints using Harris detector
                t_detKeypoints = detKeypointsHarris(keypoints, (dataBuffer.end() - 1)->cameraImg, bVisDebug);
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }
        }
        else
        {
            try
            {
                // detect keypoints using other user-specified detector types
                t_detKeypoints = detKeypointsModern(keypoints, (dataBuffer.end() - 1)->cameraImg, detectorType, bVisDebug);
            }
            catch(const char *msg)
            {
                // show error message and return 1
                cout << msg << endl;
                return 1;
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }
        }

        // store the number of detected keypoints and the processing time for keypoint detection in the last element of the result ringbuffer
        (resultBuffer.end() - 1)->numKeypoints = keypoints.size();
        (resultBuffer.end() - 1)->t_detKeypoints = t_detKeypoints;

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        // remark: This is a temporary solution that will be replace with bounding boxes provided by an object detection CNN!
        //bool bFocusOnVehicle = true;  // true: only consider keypoints within the bounding box; false: consider all keypoints
        // store bFocusOnVehicle flag in the last result ringbuffer element
        (resultBuffer.end() - 1)->bFocusOnVehicle = bFocusOnVehicle;
        cout << "Focus on keypoints on vehicle = " << bFocusOnVehicle << endl;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            // create a copy the original keypoint vector
            vector<cv::KeyPoint> keypoints_cpy;
            copy(keypoints.begin(), keypoints.end(), back_inserter(keypoints_cpy));

            // clear the contents of the original keypoint vector leaving the container with size 0
            keypoints.clear();

            // loop over all keypoints found in the current image
            for (auto keyPoint : keypoints_cpy)
            {
                // check wheter a keypoint is within the bounding box of the preceeding vehicle
                if (vehicleRect.contains(keyPoint.pt))
                {
                    // keep only those keypoints within the bounding box of the preceeding vehicle
                    keypoints.push_back(keyPoint);
                }
            }

            // viszalize keypoints before and after bounding box filtering (for debugging)
            if (bVisDebug)
            {
                // plot original keypoints (copy) before filtering
                cv::Mat visImgGray_all_Kpts = (dataBuffer.end() - 1)->cameraImg.clone();
                cv::drawKeypoints(
                    (dataBuffer.end() - 1)->cameraImg, keypoints_cpy, visImgGray_all_Kpts,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                cv::rectangle(visImgGray_all_Kpts, vehicleRect, cv::Scalar(0, 255, 0), 2, 8);
                // plot keypoints after filtering
                cv::Mat visImgGray_filtered_Kpts = (dataBuffer.end() - 1)->cameraImg.clone();
                cv::drawKeypoints(
                    (dataBuffer.end() - 1)->cameraImg, keypoints, visImgGray_filtered_Kpts,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                cv::rectangle(visImgGray_filtered_Kpts, vehicleRect, cv::Scalar(0, 255, 0), 2, 8);
                // vertically concatenate both plots
                cv::Mat visImgGray_Kpts;
                cv::vconcat(visImgGray_all_Kpts, visImgGray_filtered_Kpts, visImgGray_Kpts);
                // show concatenated plots
                string windowName_Kpts = "Keypoints before and after bounding box filtering";
                cv::namedWindow(windowName_Kpts, 5);
                cv::imshow(windowName_Kpts, visImgGray_Kpts);

                // wait for user key press
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed

                // clear temporary images and variables
                visImgGray_all_Kpts.release();
                visImgGray_filtered_Kpts.release();
                visImgGray_Kpts.release();
                windowName_Kpts.clear();
            }

            // number of keypoints befor and after filtering
            cout << "Total number of keypoints found in the overall image:    n_total  = " << keypoints_cpy.size() << endl;
            cout << "Number of keypoints in the target vehicle bounding box:  n_target = " << keypoints.size() << endl;

            // clear the copy of the original keypoint vector
            keypoints_cpy.clear();
        }

        // store the number of keypoints within the region of interest (target bounding box) in the last result ringbuffer element
        (resultBuffer.end() - 1)->numKeypointsInROI = keypoints.size();  // equal to total number of keypoints if bFocusOnVehicle == false

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful only for debugging and learning => Do not use in real application!)
        //bool bLimitKpts = false;
        (resultBuffer.end() - 1)->bLimitKpts = bLimitKpts;  // store bLimitKpts flag in the last result ringbuffer element
        cout << "Limit number of keypoints = " << bLimitKpts << endl;
        if (bLimitKpts)
        {
            // copy keypoints for debugging and visualization purpose
            vector<cv::KeyPoint> keypoints_cpy;
            copy(keypoints.begin(), keypoints.end(), back_inserter(keypoints_cpy));
        
            // int maxKeypoints = 50;  // only for testing => Do not limit keypoints in real application!

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first maxKepyoints as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            else
            {
                // sort keypoints according to the strength of the detector response (keypoints are not always sorted automatically!)
                sort(keypoints.begin(), keypoints.end(), compareKeypointResponse);

                // keep the first maxKeypoints from the list sorted by descending order of the detector response
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }            
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << "NOTE: Keypoints have been limited (n_max = " << maxKeypoints << ")!" << endl;
            cout << "The first n_max = " << keypoints.size() << " keypoints are kept. " << endl;

            // vViszalize keypoints before and after limiting their number (for debugging)
            if (bVisDebug)
            {
                // plot original keypoints (copy) before filtering
                cv::Mat visImgGray_all_Kpts = (dataBuffer.end() - 1)->cameraImg.clone();
                cv::drawKeypoints(
                    (dataBuffer.end() - 1)->cameraImg, keypoints_cpy, visImgGray_all_Kpts,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                // plot keypoints after filtering
                cv::Mat visImgGray_filtered_Kpts = (dataBuffer.end() - 1)->cameraImg.clone();
                cv::drawKeypoints(
                    (dataBuffer.end() - 1)->cameraImg, keypoints, visImgGray_filtered_Kpts,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                // vertically concatenate both plots
                cv::Mat visImgGray_Kpts;
                cv::vconcat(visImgGray_all_Kpts, visImgGray_filtered_Kpts, visImgGray_Kpts);
                // show concatenated plots
                string windowName_Kpts = "Keypoints before and after bounding box filtering";
                cv::namedWindow(windowName_Kpts, 5);
                cv::imshow(windowName_Kpts, visImgGray_Kpts);

                // wait for user key press
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed

                // clear temporary images and variables
                visImgGray_all_Kpts.release();
                visImgGray_filtered_Kpts.release();
                visImgGray_Kpts.release();
                windowName_Kpts.clear();
            }
            
            // clear the copy of the original keypoint vector
            keypoints_cpy.clear();
        }

        // store the number of limited keypoints within the region of interest (target bounding box) in the last result ringbuffer element
        (resultBuffer.end() - 1)->numKeypointsInROILimited = keypoints.size();  // equal to number of keypoints in ROI if bLimitKpts == false

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        // clear temporary variables
        keypoints.clear();

        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EVALUATE THE MEAN STRENGTH AND NEIGHBORHOOD SIZE (MEAN AND VARIANCE) OF THE REMAINING KEYPOINTS

           Note:
           - Keypoints.response = strength of the keypoint detectors's response
           - keypoints.size = keypoint diameter
           - keypoints.size() = length of keypoints vector

        */

        // calculate the mean detector response and the mean keypoint diameter over all remaining keypoints
        double meanDetectorResponse = 0.0;
        double meanKeypointDiam = 0.0;
        for (auto kPt = (dataBuffer.end() - 1)->keypoints.begin(); kPt < (dataBuffer.end() - 1)->keypoints.end(); kPt++)
        {
            meanDetectorResponse += kPt->response;
            meanKeypointDiam += kPt->size;
        }
        meanDetectorResponse /= (dataBuffer.end() - 1)->keypoints.size();
        meanKeypointDiam /= (dataBuffer.end() - 1)->keypoints.size();

        // calculate the keypoint diameter variance over all remaining keypoints
        double varianceKeypointDiam = 0.0;
        for (auto kPt = (dataBuffer.end() - 1)->keypoints.begin(); kPt < (dataBuffer.end() - 1)->keypoints.end(); kPt++)
        {
            varianceKeypointDiam += (kPt->size - meanKeypointDiam) * (kPt->size - meanKeypointDiam);
        }
        varianceKeypointDiam /= (dataBuffer.end() - 1)->keypoints.size();

        // output for debugging
        if (true) {
            cout << "meanDetectorResponse = " << meanDetectorResponse << endl;
            cout << "meanKeypointDiam = " << meanKeypointDiam << endl;
            cout << "varianceKeypointDiam = " << varianceKeypointDiam << endl;
        }

        // store the mean strength of the keypoint detector and the statistical neighborhood size in the last result ringbuffer element
        (resultBuffer.end() - 1)->meanDetectorResponse = meanDetectorResponse;
        (resultBuffer.end() - 1)->meanKeypointDiam = meanKeypointDiam;
        (resultBuffer.end() - 1)->varianceKeypointDiam = varianceKeypointDiam;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        // select keypoint descriptor extractor type
        //string descExtractorType = "BRIEF";  // options: "BRISK", "BRIEF", "ORB", "FREAK", "KAZE", "AKAZE", "SIFT", "SURF"

        // store selected keypoint descriptor type in the last result ringbuffer element
        (resultBuffer.end() - 1)->descExtractorType = descExtractorType;
        cout << "Seletect descriptor extractor tpye = " << descExtractorType << endl;

        // initialize descriptor matrix
        cv::Mat descriptors;

        // initialize processing time for keypoint descriptor extraction
        double t_descKeypoints = 0.0;

        // check if descriptor extractor and keypoint detector are

        if ( (detectorType == "SIFT") && (descExtractorType == "ORB") )
        { // skip the combination "SIFT" (keypoint detector) and "ORB" (descriptor extractor) due to memory allocation issues
            // inform user about current memory allocation problem using "SIFT" keypoint detector and "ORB"
            cout << "The keypoint detector type "
                << detectorType
                << " in combination with the descriptor extractor type "
                << descExtractorType
                << " causes memory allocation issues ..."
                << endl;
            cout << "... so this combination will be skipped" << endl;
        }
        else if (
            ((descExtractorType != "KAZE") && (descExtractorType != "AKAZE")) ||
            (((descExtractorType == "KAZE") || (descExtractorType == "AKAZE")) && ((detectorType == "KAZE") || (detectorType == "AKAZE")))
            )
        {  // "KAZE" and "AKAZE" descriptor extractors are only compatible with "KAZE" or "AKAZE" keypoints

            // extract keypoint descriptors
            try
            {
                // extract descriptors using user-specified descriptor extractor types
                t_descKeypoints = descKeypoints(
                    (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descExtractorType);
            }
            catch(const char *msg)
            {
                // show error message and return 1
                cout << msg << endl;
                return 1;
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }
            
        }
        else
        {
            // inform user about incompatibility of descriptor extractor and keypoint detector type
            cout << "No descriptor extraction possible ... as descriptor extractor type "
                << descExtractorType
                << " is not compatible with keypoint detector type "
                << detectorType
                << endl;
        }

        // store the processing time for keypoint descriptor extration in the last result ringbuffer element
        (resultBuffer.end() - 1)->t_descKeypoints = t_descKeypoints;

        // store the cumulated processing time for keypoint detection and descriptor extraction in the last result ringbuffer element
        (resultBuffer.end() - 1)->t_sum_det_desc = t_detKeypoints + t_descKeypoints;

        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        // clear temporary variables
        descriptors.release();

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        /* MATCH KEYPOINT DESCRIPTORS */

        // Initialize processing time for keypoint descriptor matching
        double t_matchDescriptors = 0.0;

        // Match keypoint descriptors
        if (dataBuffer.size() > 1)  // wait until at least two images have been processed
        {
            // check if descriptor extractor and keypoint detector are compatible
            if ( (detectorType == "SIFT") && (descExtractorType == "ORB") )
            { // skip the combination "SIFT" (keypoint detector) and "ORB" (descriptor extractor) due to memory allocation issues
                // inform user about current memory allocation problem using "SIFT" keypoint detector and "ORB"
                cout << "The keypoint detector type "
                    << detectorType
                    << " in combination with the descriptor extractor type "
                    << descExtractorType
                    << " causes memory allocation issues ..."
                    << endl;
                cout << "... so this combination will be skipped" << endl;

                // store configuration in result buffer
                (resultBuffer.end() - 1)->matcherType = matcherType;     // store selected matcher type in result buffer
                (resultBuffer.end() - 1)->descriptorType = descriptorType;  // store selected descriptor type in result buffer
                (resultBuffer.end() - 1)->selectorType = selectorType;  // store selected selector type in result buffer
                (resultBuffer.end() - 1)->numDescMatches = -1;  // store number of descriptor matches in result buffer
            }
            else if (
                ((descExtractorType != "KAZE") && (descExtractorType != "AKAZE")) ||
                (((descExtractorType == "KAZE") || (descExtractorType == "AKAZE")) && ((detectorType == "KAZE") || (detectorType == "AKAZE")))
                )
            { // "KAZE" and "AKAZE" descriptor extractors are only compatible with "KAZE" or "AKAZE" keypoints

                // create vector of keypoint descriptor matches
                vector<cv::DMatch> matches;
                //string matcherType = "MAT_BF";            // MAT_BF, MAT_FLANN
                //string descriptorType = "DES_BINARY";     // DES_BINARY, DES_HOG
                //string selectorType = "SEL_KNN";          // SEL_NN, SEL_KNN

                // store selected matcher type in the last result ringbuffer element
                (resultBuffer.end() - 1)->matcherType = matcherType;
                cout << "Seletect descriptor matcher tpye = " << matcherType << endl;

                // store selected descriptor type in the last result ringbuffer element
                (resultBuffer.end() - 1)->descriptorType = descriptorType;
                cout << "Seletect descriptor tpye = " << descriptorType << endl;

                // store selected selector type in the last result ringbuffer element
                (resultBuffer.end() - 1)->selectorType = selectorType;
                cout << "Seletect selector tpye = " << selectorType << endl;

                //// STUDENT ASSIGNMENT
                //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                // print out which images are used
                cout << "Matching keypoint descriptors between the last and the second last image stored in the ringbuffer:" << endl;
                cout << "Filename of last image in ringbuffer     = " << (dataBuffer.end() - 1)->imgFilename << endl;
                cout << "Filename of 2nd last image in ringbuffer = " << (dataBuffer.end() - 2)->imgFilename << endl;

                try
                {
                    // match keypoint descriptors between the last and the second last image stored in the ringbuffer
                    t_matchDescriptors = matchDescriptors(
                        (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                        (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                        matches, descriptorType, matcherType, selectorType);
                }
                catch(const exception& e)
                {
                    // show exeption and return 1
                    cerr << e.what() << endl;
                    return 1;
                }

                //// EOF STUDENT ASSIGNMENT

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                // store number of matches in the last result ringbuffer element
                (resultBuffer.end() - 1)->numDescMatches = matches.size();

                // clear temporary variables
                matches.clear();

                cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                // visualize matches between the current and the previous image
                // bVis = true;
                if (bVis)
                {
                    // plot keypoint matches between the current and the previous image
                    cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                    (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 1)->kptMatches, matchImg,
                                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    string windowName = "Matching keypoints between two camera images";
                    cv::namedWindow(windowName, 7);
                    cv::imshow(windowName, matchImg);

                    // wait for user key press
                    cout << "Press key to continue to next image" << endl;
                    cv::waitKey(0);  // wait for key to be pressed

                    // clear temporary images and variables
                    matchImg.release();
                    windowName.clear();
                }
                // bVis = false;

            }
            else
            {
                // inform user about incompatibility of "KAZE" descriptor extractor with keypoint detectors other than "KAZE" or "AKAZE"
                cout << "Descriptor extractor type "
                    << descExtractorType
                    << " is not compatible with keypoint detector type "
                    << detectorType
                    << endl;

                // store configuration in result buffer
                (resultBuffer.end() - 1)->matcherType = matcherType;     // store selected matcher type in result buffer
                (resultBuffer.end() - 1)->descriptorType = descriptorType;  // store selected descriptor type in result buffer
                (resultBuffer.end() - 1)->selectorType = selectorType;  // store selected selector type in result buffer
                (resultBuffer.end() - 1)->numDescMatches = -1;  // store number of descriptor matches in result buffer
            }
        }
        else
        {
            // store empty strings for the first image in result buffer
            (resultBuffer.end() - 1)->matcherType = "";     // store selected matcher type in result buffer
            (resultBuffer.end() - 1)->descriptorType = "";  // store selected descriptor type in result buffer
            (resultBuffer.end() - 1)->selectorType = "";    // store selected selector type in result buffer
            (resultBuffer.end() - 1)->numDescMatches = 0;   // store number of descriptor matches in result buffer
        }

        // store processing time for keypoint descriptor matching in the last result ringbuffer element
        (resultBuffer.end() - 1)->t_matchDescriptors = t_matchDescriptors; // is always zero for the first image

        // store the cumulated processing time for keypoint detection, descriptor extraction and descriptor matching 
        // in the last result ringbuffer element
        (resultBuffer.end() - 1)->t_sum_det_desc_match = t_detKeypoints + t_descKeypoints + t_matchDescriptors;

    } // eof loop over all images

    // push current result buffer into the vector of result buffers
    evalResultBuffers.push_back(resultBuffer);

    /*
    // debugging
    cout << "Test access of last element in evalResultBuffers:" << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->detectorType << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->descExtractorType << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->matcherType << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->descriptorType << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->selectorType << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->numKeypoints << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->numKeypointsInROI << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->numKeypointsInROILimited << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->numDescMatches << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->t_detKeypoints << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->t_descKeypoints << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->t_sum_det_desc << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->t_matchDescriptors << endl;
    cout << ((*(evalResultBuffers.end() - 1)).end() - 1)->t_sum_det_desc_match << endl;

    cout << "Current size of evalResultsBuffers = " << evalResultBuffers.size() << endl;

    cout << "Press any key to continue ...";
    cin.get();
    */

    if (bExportResultsToCSV)
    {
        // define filepath and filename for result file
        string resultFilepath = dataPath + "results/";
        string resultFilename = "2D_feature_tracking_using_";
        string resultFiletype = ".csv";
        string resultFullFilename = resultFilepath
                                + resultFilename
                                + resultBuffer.begin()->detectorType
                                + "_and_"
                                + resultBuffer.begin()->descExtractorType
                                + resultFiletype;
        cout << "Export evaluation results to " << resultFullFilename << endl;

        // try to export the results to csv file
        try
        {
            // export results to csv file
            exportResultsToCSV(resultFullFilename, resultBuffer);
        }
        catch(const exception& e)
        {
            // show exeption and return 1
            cerr << e.what() << endl;
            return 1;
        }

        cout << "#5 : EXPORT RESULTS TO CSV FILE done" << endl;
    }

    // return 0 if program terminates without errors
    return 0;
}


/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* SWITCH BETWEEN SINGLE RUN (USING MANUAL CONFIGURATION) AND BATCH RUN (EVALUATING DIFFERENT DETECTOR EXTRACTOR COMBINATIONS) */

    // Select batch run or single run mode
    bool BatchMode = false; // options: true => batch run; false => single run

    if (!BatchMode)
    {
        /* --- SINGLE RUN MODE --- */

        // print selected run mode
        cout << "Evaluation of 2D feature tracking in single run mode:" << endl;
        
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        
        // ringbuffer implementation based on boost circular buffer, s. fun_2DFeatureTracking()
        
        // check installed boost version
        cout << "Using Boost version "
            << BOOST_VERSION / 100000     << "."  // major version
            << BOOST_VERSION / 100 % 1000 << "."  // minor version
            << BOOST_VERSION % 100                // patch level
            << endl;
        
        /* MANUAL CONFIGURATION FOR 2D FEATURE TRACKING STUDENT ASSIGNMENT */

        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        // Select keypoint detector type
        string detectorType = "FAST";  // options: "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "KAZE", "AKAZE", "SIFT", "SURF"

        //// TASK MP.3 -> only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;  // true: only consider keypoints within the bounding box; false: consider all keypoints
        
        // optional : limit number of keypoints (helpful only for debugging and learning => Do not use in real application!)
        bool bLimitKpts = false;
        int maxKeypoints = 50;  // only for testing => Do not limit keypoints in real application!
        
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
        
        // select keypoint descriptor extractor type ("KAZE" and "AKAZE" only work with "KAZE" or "AKAZE" keypoints)
        string descExtractorType = "BRIEF";  // options: "BRISK", "BRIEF", "ORB", "FREAK", "KAZE", "AKAZE", "SIFT", "SURF"
        
        //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
        //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t = 0.8 in file matching2D.cpp

        // select descriptor matcher tpye
        string matcherType = "MAT_BF";  // options: MAT_BF, MAT_FLANN

        // select descriptor type (use "DES_HOG" for "KAZE", "SIFT" and "SURF", otherwise use "DES_BINARY")
        string descriptorType = "DES_BINARY";  // options: DES_BINARY, DES_HOG
        
        // select selector type
        string selectorType = "SEL_KNN";  // SEL_NN, SEL_KNN
        
        // result visualization
        bool bVis = true;        // visualization of keypoint matching results
        bool bVisDebug = false;  // visualize intermediate results (for debugging or to look into details)
        
        // export evaluation results to csv file
        bool bExportResultsToCSV = true;
        
        // initialize (empty) vector of evaluation result buffers
        vector<boost::circular_buffer<EvalResults>> evalResultBuffers;
        
        // try to run 2D feature tracking in single run mode
        try
        {
            // evaluate 2D feature tracking performance
            if (fun_2DFeatureTracking(
                evalResultBuffers,
                detectorType,
                bFocusOnVehicle,
                bLimitKpts,
                maxKeypoints,
                descExtractorType,
                matcherType,
                descriptorType,
                selectorType,
                bVis,
                bVisDebug,
                bExportResultsToCSV) == 0)
            {
                // return 0 if program terminates without errors
                return 0;
            }
            else
            {
                // return 1 if program terminates with errors
                return 1;
            }
        }
        catch(const exception& e)
        {
            // show exeption and return 1
            cerr << e.what() << endl;
            return 1;
        }

    }
    else
    {
        /* --- BATCH RUN MODE --- */

        // print selected run mode
        cout << "Evaluation of 2D feature tracking in batch run mode:" << endl;
        
        // vector of keypoint detector types to evaluate
        vector<string> vec_detectorTypes = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "KAZE", "AKAZE", "SIFT", "SURF"};

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        
        // optional : limit number of keypoints (helpful only for debugging and learning => Do not use in real application!)
        bool bLimitKpts = false;
        int maxKeypoints = 100;
        
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
        
        // vector of keypoint descriptor extractor types to evaluate
        vector<string> vec_descExtractorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "KAZE", "AKAZE", "SIFT", "SURF"};
        
        //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
        //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t = 0.8 in file matching2D.cpp

        // set descriptor matcher tpye
        string matcherType = "MAT_BF";  // options: MAT_BF, MAT_FLANN

        // set binary descriptor type => will be automatically adapted to the descriptor extractor type in the loop over all combinations
        string descriptorType = "DES_BINARY";  // options: DES_BINARY, DES_HOG    
        
        // set selector type
        string selectorType = "SEL_KNN";  // SEL_NN, SEL_KNN
        
        // result visualization
        bool bVis = false;       // visualization of keypoint matching results
        bool bVisDebug = false;  // visualize intermediate results (for debugging or to look into details)
        
        // export evaluation results to csv file
        bool bExportResultsToCSV = true;
        
        // vector of evaluation result buffers
        //int evalResultBuffersSize = 1;
        //vector<boost::circular_buffer<EvalResults>> evalResultBuffers(evalResultBuffersSize);
        vector<boost::circular_buffer<EvalResults>> evalResultBuffers;

        // iterator
        int itr = 0;

        for (vector<string>::const_iterator ptrDetType = vec_detectorTypes.begin(); ptrDetType != vec_detectorTypes.end(); ptrDetType++)
        {

            for (vector<string>::const_iterator ptrDescExtType = vec_descExtractorTypes.begin(); ptrDescExtType != vec_descExtractorTypes.end(); ptrDescExtType++)
            {

                // print current iterator
                cout << "\nIteration no. " << itr++ << endl;

                if ((*ptrDescExtType) == "KAZE" || (*ptrDescExtType) == "SIFT" || (*ptrDescExtType) == "SURF")
                {
                    // use gradient based descriptor type for SIFT and SURF
                    descriptorType = "DES_HOG";
                }
                else
                {
                    // use binary descriptor type for all other descriptor extractor types
                    descriptorType = "DES_BINARY";
                }
                
                // print current configuration
                cout << "\n" << "Next configuration for 2D feature tracking:" << endl;
                cout << "Feature detector type     = " << (*ptrDetType) << endl;
                cout << "Descriptor extractor type = " << (*ptrDescExtType) << endl;
                cout << "Matcher type              = " << matcherType << endl;
                cout << "Descriptor type           = " << descriptorType << endl;
                cout << "Selector type             = " << selectorType << "\n" << endl;

                // KADZE and AKAZE feature extractors only work with KAZE or AKAZE keypoints
                // => skip other configurations

                // try to run 2D feature tracking in batch run mode
                try
                {
                    // evaluate 2D feature tracking performance in batch mode
                    if (fun_2DFeatureTracking(
                        evalResultBuffers,
                        (*ptrDetType),
                        bFocusOnVehicle,
                        bLimitKpts,
                        maxKeypoints,
                        (*ptrDescExtType),
                        matcherType,
                        descriptorType,
                        selectorType,
                        bVis,
                        bVisDebug,
                        bExportResultsToCSV) == 0)
                    {
                        continue;
                    }
                    else
                    {
                        // return 1 if program terminates with errors
                        return 1;
                    }
                }
                catch(const exception& e)
                {
                    // show exeption and return 1
                    cerr << e.what() << endl;
                    return 1;
                }

                // wait for user key press
                string tmp;
                cout << "Press any key to continue: ";
                cin >> tmp;
                cout << "endl";

            }

        }

        // export overall results in an overview on all keypoint detector - descriptor extractor combinations to a csv file
        if (bExportResultsToCSV)
        {
            // define filepath and filename for result file
            string resultFilepath = dataPath + "results/";
            string resultFilename = "2D_feature_tracking_overall_results";
            string resultFiletype = ".csv";
            string resultFullFilename = resultFilepath
                                    + resultFilename
                                    + resultFiletype;
            cout << "Export overall evaluation results to " << resultFullFilename << endl;

            // try to export the results to csv file
            try
            {
                // export overall results to csv file
                exportOverallResultsToCSV(resultFullFilename, evalResultBuffers);
            }
            catch(const exception& e)
                {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }

            cout << "#6 : EXPORT OVERALL RESULTS TO CSV FILE done" << endl;

        }

        // return 0 if program terminates without errors
        return 0;
        
    }

}
