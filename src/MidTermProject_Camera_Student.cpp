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

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera data frames
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000";  // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // check installed boost version
    cout << "Using Boost version "
        << BOOST_VERSION / 100000     << "."  // major version
        << BOOST_VERSION / 100 % 1000 << "."  // minor version
        << BOOST_VERSION % 100                // patch level
        << endl;

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

    // visualization 
    bool bVis = true;        // visualize results
    bool bVisDebug = false;  // visualize intermediate results (for debugging or to look into details)

    // circular data buffer to hold evaluation results
    bool bExportResultsToCSV = true;
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

        // push camera image and filename into (circular) data frame buffer
        DataFrame frame;
        frame.imgFilename = imgNumber.str() + imgFileType;  // store filename without path
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        // create structure to hold the evaluation results
        EvalResults results;
        results.imgFilename = imgNumber.str() + imgFileType;  // store filename without path
        
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
        string detectorType = "HARRIS";  // options: "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "KAZE", "AKAZE", "SIFT", "SURF"
        results.detectorType = detectorType;  // store selected detector type in results
        cout << "Seletect keypoint detector tpye = " << detectorType << endl;

        // create empty feature list for current image
        vector<cv::KeyPoint> keypoints;

        // Initialize processing time for keypoint detection
        double t_detKeypoints = 0.0;
        
        // detect keypoints
        if (detectorType.compare("SHITOMASI") == 0)
        {
            // Detect keypoints using Shi-Tomasi detector
            t_detKeypoints = detKeypointsShiTomasi(keypoints, imgGray, bVisDebug);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            // detect keypoints using Harris detector
            t_detKeypoints = detKeypointsHarris(keypoints, imgGray, bVisDebug);
        }
        else
        {
            try
            {
                // detect keypoints using other user-specified detector types
                t_detKeypoints = detKeypointsModern(keypoints, imgGray, detectorType, bVisDebug);
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

        // store the number of all detected keypoints in the image and the processing time for keypoint detection in results
        results.numKeypoints = keypoints.size();
        results.t_detKeypoints = t_detKeypoints;

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        // remark: This is a temporary solution that will be replace with bounding boxes provided by an object detection CNN!
        bool bFocusOnVehicle = true;  // true: only consider keypoints within the bounding box; false: consider all keypoints
        results.bFocusOnVehicle = bFocusOnVehicle; // store bFocusOnVehicle flag in results
        cout << "Focus on keypoints on vehicle = " << bFocusOnVehicle << endl;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            // create an empty feature list for the filtered keypoints of the current image
            vector<cv::KeyPoint> keypoints_filtered;

            // loop over all keypoints found in the current image
            for (auto keyPoint : keypoints)
            {
                // check wheter a keypoint is within the bounding box of the preceeding vehicle
                if (vehicleRect.contains(keyPoint.pt))
                {
                    // keep only those keypoints within the bounding box of the preceeding vehicle
                    keypoints_filtered.push_back(keyPoint);
                }
            }

            // viszalize keypoints before and after bounding box filtering (for debugging)
            if (bVisDebug)
            {
                // plot keypoints before filtering
                cv::Mat visImgGray_all_Kpts = imgGray.clone();
                cv::drawKeypoints(
                    imgGray, keypoints, visImgGray_all_Kpts,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                cv::rectangle(visImgGray_all_Kpts, vehicleRect, cv::Scalar(0, 255, 0), 2, 8);
                // plot keypoints after filtering
                cv::Mat visImgGray_filtered_Kpts = imgGray.clone();
                cv::drawKeypoints(
                    imgGray, keypoints_filtered, visImgGray_filtered_Kpts,
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
            cout << "Total number of keypoints found in the image: n_total  = "<< keypoints.size() << endl;
            cout << "Number of keypoints in the target bound box:  n_target = " << keypoints_filtered.size() << endl;

            // replace original feature list with filtered feature list
            keypoints = keypoints_filtered;

            // print keypoint filtering results
            cout << "Focus on n = " << keypoints.size() << " keypoints located in target vehicle bounding box. " << endl;
        }

        // store the number of keypoints within the region of interest resp. the target bounding box in results
        results.numKeypointsInROI = keypoints.size();  // is equal to the total number of keypoints if bFocusOnVehicle == false

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful only for debugging and learning => Do not use in real application!)
        bool bLimitKpts = false;
        results.bLimitKpts = bLimitKpts;  // store bLimitKpts flag in results
        cout << "Limit number of keypoints = " << bLimitKpts << endl;
        if (bLimitKpts)
        {
            // copy keypoints for debugging and visualization purpose
            vector<cv::KeyPoint> keypoints_cpy;
            copy(keypoints.begin(), keypoints.end(), back_inserter(keypoints_cpy));
        
            int maxKeypoints = 50;  // only for testing => Do not limit keypoints in real application!

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
                // plot keypoints before filtering
                cv::Mat visImgGray_all_Kpts = imgGray.clone();
                cv::drawKeypoints(
                    imgGray, keypoints_cpy, visImgGray_all_Kpts,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                // plot keypoints after filtering
                cv::Mat visImgGray_filtered_Kpts = imgGray.clone();
                cv::drawKeypoints(
                    imgGray, keypoints, visImgGray_filtered_Kpts,
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
                keypoints_cpy.clear();
            }
        }

        // store the number of limited keypoints within the region of interest resp. the target bounding box in results
        results.numKeypointsInROILimited = keypoints.size();  // is equal to the number of keypoints within ROI if bLimitKpts == false

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        // select keypoint descriptor extractor type
        string descExtractorType = "BRIEF";  // options: "BRISK", "BRIEF", "ORB", "FREAK", "KAZE", "AKAZE", "SIFT", "SURF"
        results.descExtractorType = descExtractorType;  // store selected keypoint descriptor type in results
        cout << "Seletect descriptor extractor tpye = " << descExtractorType << endl;

        // initialize descriptor matrix
        cv::Mat descriptors;

        // initialize processing time for keypoint descriptor extraction
        double t_descKeypoints = 0.0;

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

        // store the processing time for keypoint descriptor extration in results
        results.t_descKeypoints = t_descKeypoints;

        // store the cumulated processing time for keypoint detection and descriptor extraction in results
        results.t_sum_det_desc = t_detKeypoints + t_descKeypoints;

        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        // Initialize processing time for keypoint descriptor matching
        double t_matchDescriptors = 0.0;  

        // Match keypoint descriptors
        if (dataBuffer.size() > 1)  // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";            // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY";     // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";          // SEL_NN, SEL_KNN
            results.matcherType = matcherType;        // store selected matcher type in results
            results.descriptorType = descriptorType;  // store selected descriptor type in results
            results.selectorType = selectorType;      // store selected selector type in results
            cout << "Seletect descriptor matcher tpye = " << matcherType << endl;
            cout << "Seletect descriptor tpye = " << descriptorType << endl;
            cout << "Seletect selector tpye = " << selectorType << endl;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            // print out which images are used
            cout << "Matching keypoint descriptors between the last and the second last image stored in the ringbuffer:" << endl;
            cout << "Filename of last image in ringbuffer     = " << (dataBuffer.end() - 1)->imgFilename << endl;
            cout << "Filename of 2nd last image in ringbuffer = " << (dataBuffer.end() - 2)->imgFilename << endl;
            
            // match keypoint descriptors between the last and the second last image stored in the ringbuffer
            t_matchDescriptors = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                                (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                                matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            // store number of matches in results
            results.numDescMatches = matches.size();

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                // wait for user key press
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0);  // wait for key to be pressed
            }
            bVis = false;
        }
        else
        {
            // store empty strings for the first imgae
            results.matcherType = "";     // store selected matcher type in results
            results.descriptorType = "";  // store selected descriptor type in results
            results.selectorType = "";    // store selected selector type in results
            results.numDescMatches = 0;   // store number of descriptor matches in results
        }
        

        // store processing time for keypoint descriptor matching in results
        results.t_matchDescriptors = t_matchDescriptors; // is always zero for the first image

        // store the cumulated processing time for keypoint detection, descriptor extraction and descriptor matching in results
        results.t_sum_det_desc_match = t_detKeypoints + t_descKeypoints + t_matchDescriptors;

        // push current evaluation results into the tail of the circular result buffer
        resultBuffer.push_back(results);

    } // eof loop over all images

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
