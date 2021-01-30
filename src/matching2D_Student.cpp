// ============================================================================
//  
//  Project 2.1: 2D Feature Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_2D_Feature_Tracking
//
// ============================================================================

// function definitions for 2D keypoint detection and 2D feature matching

#include <numeric>
#include "matching2D.hpp"

using namespace std;


// find best matches for keypoints in two camera images based on several matching methods
double matchDescriptors(
    std::vector<cv::KeyPoint> & kPtsSource, std::vector<cv::KeyPoint> & kPtsRef, cv::Mat & descSource, cv::Mat & descRef,
    std::vector<cv::DMatch> & matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{

    // init and configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    { // use brute-force matching

        // select ditance norm type to compare the descriptors
        //int normType = cv::NORM_HAMMING;
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;

        // brute force matching approach searching through all avaialable keypoints and keypoint descriptors 
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "Use BF matching: BF cross-check = " << crossCheck << endl;

    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    { // use FLANN-based matching

        // workaround for BUG in OpenCV
        if (descSource.type() != CV_32F || descRef.type()!=CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        // efficient FLANN-based matching using a KD-tree to quickly search through available keypoints and descriptors
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "Use FLANN matching" << endl;

    }

    // create variable to hold the processing time for keypoint descriptor matching
    double t = 0.0;

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        // perform nearest neighbor matching: yields only one best match (timed process)
        t = (double)cv::getTickCount(); // trigger timer
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // stop timer
        cout << " NN matching with n = "
            << matches.size()
            << " matches in "
            << 1000 * t / 1.0
            << " ms"
            << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // perform k nearest neighbor matching: yields k best matches, here: k = 2 (timed process)
        vector<vector<cv::DMatch>> knn_matches;
        t = (double)cv::getTickCount(); // trigger timer
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the two (k = 2) best matches
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // stop timer
        cout << "KNN matching with n = " << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        /* NOTE: 
           Descriotor distance ratio test versus cross-check matching:
           In general, the descriotor distance ratio test is less precise, but more efficient than cross-check
           matching. 
           In cross-check matching, the keypoint matching is done twice (image1 -> image 2 and vice versa), and
           keypoints are only accepted when keypoint matches found in image 2 for keypoints from image 1 match
           with keypoint matches found in image 1 for keypoints from image 2. As this needs more processing time
           the distance ratio test is preferred in this task.
        */

        // filter out ambiguous matches using descriptor distance ratio test and reduce the number of false positives
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "Total number of keypoints matches = " << matches.size() << endl;
        cout << "Number of keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }

    // return processing time for keypoint descriptor matching
    return t;

}


// use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(
    vector<cv::KeyPoint> & keypoints, cv::Mat & img, cv::Mat & descriptors, string descExtractorType)
{

    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descExtractorType.compare("BRISK") == 0)
    {
        // BRISK (Binary robust invariant scalable keypoints) feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/de/dbf/classcv_1_1BRISK.html
        
        // set BRISK descriptor extractor parameters
        int threshold = 30;  // FAST/AGAST detection threshold score
        int octaves = 3;  // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f;  // apply this scale to the pattern used for sampling the neighbourhood of a keypoint

        // create BRISK descriptor extractor
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descExtractorType.compare("BRIEF") == 0)
    {
        // BRIEF (Binary robust independent elementary features)
        // OpenCV: https://docs.opencv.org/4.1.2/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html
        
        // set BRIEF descriptor extractor parameters
        int bytes = 32;  // length of the descriptor in bytes, valid values are: 16, 32 (default) or 64
		bool use_orientation = false;  // sample patterns using keypoints orientation, disabled by default

        // create BRIEF descriptor extractor
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (descExtractorType.compare("ORB") == 0)
    {
        // ORB (Oriented FAST and Rotated BRIEF)
        // OpenCV: https://docs.opencv.org/4.1.2/db/d95/classcv_1_1ORB.html
        
        // set ORB descriptor extractor parameters
        int nfeatures = 500;  // maximum number of features to retain
		float scaleFactor = 1.2f;  // pyramid decimation ratio, greater than 1 (scaleFactor==2 => classical pyramid)
        int nlevels = 8;  // number of pyramid levels (smallest level has linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel))
		int edgeThreshold = 31;  // size of the border where the features are not detected (should roughly match the patchSize parameter)
		int firstLevel = 0;  // level of pyramid to put source image to (Previous layers are filled with upscaled source image)
		int WTA_K = 2;  // number of points that produce each element of the oriented BRIEF descriptor (default value: 2, other possible values: 3, 4)
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;  // default: use HARRIS_SCORE to rank features (or: FAST_SCORE => slightly less stable keypoints, but faster)
		int patchSize = 31;  // size of the patch used by the oriented BRIEF descriptor
		int fastThreshold = 20;  // the fast threshold

        // create ORB descriptor extractor
		extractor = cv::ORB::create(
            nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
            WTA_K, scoreType, patchSize, fastThreshold);
	}
    else if (descExtractorType.compare("FREAK") == 0)
    {
        // FREAK (Fast Retina Keypoint) descriptor extractor
        // https://docs.opencv.org/4.1.2/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html
        // Remark: FREAK is only a keypoint descriptor! => Another feature detector is needed to find the keypoints.

        // set FREAK descriptor extractor parameters
        bool orientationNormalized = true;  // enable orientation normalization
		bool scaleNormalized = true;  // enable scale normalization
		float patternScale = 22.0f;  // scaling of the description pattern
		int nOctaves = 4;  // number of octaves covered by the detected keypoints
		// const std::vector< int > & selectedPairs = std::vector< int >();  // (optional) user defined selected pairs indexes

        // create FREAK descriptor extractor
        extractor = cv::xfeatures2d::FREAK::create(
            orientationNormalized, scaleNormalized, patternScale, nOctaves);
    }
    else if (descExtractorType.compare("KAZE") == 0)
	{
        // KAZE feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/d3/d61/classcv_1_1KAZE.html

        // set KAZE descriptor extractor parameters
        bool extended = false;  // set to enable extraction of extended (128-byte) descriptor.
		bool upright = false;  // set to enable use of upright descriptors (non rotation-invariant).
		float threshold = 0.001f;  // detector response threshold to accept point
		int nOctaves = 4;  // maximum octave evolution of the image
		int nOctaveLayers = 4;  // Default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;  // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER

        // create KAZE descriptor extractor
        extractor = cv::KAZE::create(
            extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity); 	
	}
	else if (descExtractorType.compare("AKAZE") == 0)
	{
        // AKAZE (Accelerated-KAZE) feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/d8/d30/classcv_1_1AKAZE.html
        
        // set AKAZE descriptor extractor parameters
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;  // options: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
		int descriptor_size = 0;  // size of the descriptor in bits. 0 -> full size
		int descriptor_channels = 3; // number of channels in the descriptor (1, 2, 3)
		float threshold = 0.001f;  // detector response threshold to accept point
		int nOctaves = 4;  // maximum octave evolution of the image
		int nOctaveLayers = 4;  // Default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;  // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        
        // create AKAZE descriptor extractor
        extractor = cv::AKAZE::create(
            descriptor_type, descriptor_size, descriptor_channels, threshold,
            nOctaves, nOctaveLayers, diffusivity);
	}
    else if (descExtractorType.compare("SIFT") == 0)
    {
        // SIFT (Scale Invariant Feature Transform) feature detector and descriptor extractor
        // https://docs.opencv.org/4.1.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
        
        // set SIFT descriptor extractor parameters
        int nfeatures = 0;  // number of best features to retain (features are ranked by their scores measured as the local contrast)
        int nOctaveLayers = 3;  // number of layers in each octave (3 is the value used in D. Lowe paper)
        double contrastThreshold = 0.04;  // contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        double edgeThreshold = 10;  // threshold used to filter out edge-like features
        double sigma = 1.6;  // sigma of the Gaussian applied to the input image at the octave #0

        // create SIFT descriptor extractor
		//extractor = cv::xfeatures2d::SiftDescriptorExtractor::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    	extractor = cv::xfeatures2d::SIFT::create(
            nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else if (descExtractorType.compare("SURF") == 0)
    {
        // SURF (Speeded-up robust features) feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html#details

        // set SURF descriptor extractor parameters
        double hessianThreshold = 100;  // threshold for hessian keypoint detector used in SURF
		int nOctaves = 4;  // number of pyramid octaves the keypoint detector will use
		int nOctaveLayers = 3;  // number of octave layers within each octave
		bool extended = false;  // extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors)
		bool upright = false;  // up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation)

        // create SURF descriptor extractor
        //extractor = cv::xfeatures2d::SurfDescriptorExtractor::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
        extractor = cv::xfeatures2d::SURF::create(
            hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    }
    else {
        // throw error message
        throw "Error: Wrong input argument to descKeypoints(): Feature descriptor (extractor) type not defined!";
	}

    // perform feature description (timed process)
    double t = (double)cv::getTickCount();  // trigger timer
    extractor->compute(img, keypoints, descriptors);  // extract feature descriptors
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();  // stop timer
    cout << descExtractorType
        << " descriptor extraction for n = "
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

    // return processing time for keypoint descriptor extraction
    return t;
}


// detect keypoints in image using the traditional Shi-Tomasi corner detector (based in image gradients, slow)
double detKeypointsShiTomasi(vector<cv::KeyPoint> & keypoints, cv::Mat & img, bool bVis)
{

    // Shi-Tomasi corner detector
    // OpenCV: https://docs.opencv.org/4.1.2/d8/dd8/tutorial_good_features_to_track.html

    // compute Shi-Tomasi detector parameters based on image size
    int blockSize = 4;  //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0;  // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;  // minimum possible Euclidean distance between the returned corners
    int maxCorners = img.rows * img.cols / max(1.0, minDistance);  // max. num. of keypoints
    bool useHarrisDetector = false;  // parameter indicating whether to use a Harris detector or cornerMinEigenVal
    double qualityLevel = 0.01;  // minimal accepted quality of image corners
    double k = 0.04;  // free parameter of the Harris detector

    // apply corner detection
    double t = (double)cv::getTickCount();  // trigger timer
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();  // stop timer
    cout << "Shi-Tomasi feature detection with n="
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

    // visualize results
    if (bVis)
    {
        // plot image with keypoints
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);

        // wait for user key press
        cv::waitKey(0);

        // clear temporary images and variables
        visImage.release();
        windowName.clear();
    }

    // return processing time for keypoint detection
    return t;

}


// detect keypoints in image using the traditional Harris corner detector
double detKeypointsHarris(vector<cv::KeyPoint> & keypoints, cv::Mat & img, bool bVis)
{

    // Harris corner detector
    // OpenCV: https://docs.opencv.org/4.1.2/d4/d7d/tutorial_harris_detector.html
    //         https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html

    // set Harris corner detector parameters
    int blockSize = 2;  // for every pixel, a blockSize × blockSize neighborhood is considered for corner detection
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100;  // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;  // Harris parameter (see equation for details)

    // apply Harris corner detector and normalize output
    double t = (double)cv::getTickCount();  // trigger timer
    cv::Mat dst, dst_norm, dst_norm_scaled;  // define destination matrices for intermediate and final results
    dst = cv::Mat::zeros(img.size(), CV_32FC1);  // initialize final result matrix
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);  // detect Harris corners
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());  // normalize input array
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);  // scales, calculates absolute values, and converts the result to 8-bit

    // Locate local maxima in Harris corner detector response map and perform non-maximum suppression 
    // in the local neighborhood around each maximum and store the resulting keypoint coordinates in a
    // list of keypoints of type vector<cv::KeyPoint> (s. input argument)

    // set maximum permissible overlap between two features in %, used during non-maxima suppression
    double maxOverlap = 0.0;

    // loop over all rows and colums of Harris corner detector response map
    for (size_t j = 0; j < dst_norm.rows; j++)
    { // loop over all rows
    
        for (size_t i = 0; i < dst_norm.cols; i++)
        {  // loop over all cols

            // get response from normalized Harris corner detection response matrix scaled to positive 8 bit values
            int response = (int)dst_norm.at<float>(j, i);

            // only store points above a required minimum threshold (s. Harris detector parameters)
            if (response > minResponse)
            {
                // create new keypoint from keypoint (corner) detector response at the current image location (i, j)
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);   // keypoint coordinates
                newKeyPoint.size = 2 * apertureSize;  // keypoint diameter (region of interest) = 2 * Sobel filter aperture size
                newKeyPoint.response = response;      // keypoint detector response

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                { // loop over all previous keypoints found so far
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it); // get current keypoint overlap
                    if (kptOverlap > maxOverlap)
                    { // if keypoints overlap check which one contains a stronger response and keep the largest
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                       // if overlap is > t AND response is higher for new kpt
                            *it = newKeyPoint;  // replace old key point with new one
                            break;              // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                { // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint);  // store new keypoint in dynamic list
                }
            }

        } // eof loop over cols

    } // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();  // stop timer
    cout << "Harris feature detection with n="
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

    // visualize results
    if (bVis)
    {
        // plot image with keypoints
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);

        // wait for user key press
        cv::waitKey(0);

        // clear temporary images and variables
        visImage.release();
        windowName.clear();
    }

    // return processing time for keypoint detection
    return t;

}


// detect keypoints using different newer feature detectors from OpenCV like FAST, BRISK, ...) ... except 
// for the traditional Shi-Tomasi and Harris detectors, which are implemented separately (s. above)
double detKeypointsModern(
    std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, std::string detectorType, bool bVis)
{
    
    // initialize feature detector
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
	{	
        // FAST feature detector
        // OpenCV: https://docs.opencv.org/4.1.2/df/d74/classcv_1_1FastFeatureDetector.html
        
        // set FAST feature detector parameters
        // int threshold=30;
        // int threshold=20;
        int threshold=10;  // threshold on difference between intensity of the central pixel and pixels of a circle around this pixel
        bool nonmaxSuppression=true;  // if true, non-maximum suppression is applied to detected corners (keypoints)
        cv::FastFeatureDetector::DetectorType type=cv::FastFeatureDetector::TYPE_9_16;  // neighborhoods: TYPE_5_8, TYPE_7_12, TYPE_9_16

        // create FAST feature detector
        detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
	}
	else if (detectorType.compare("BRISK") == 0)
	{
        // BRISK (Binary robust invariant scalable keypoints) feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/de/dbf/classcv_1_1BRISK.html
        
        // set BRISK feature detector parameters
        // int threshold = 60;
        int threshold = 30;  // AGAST detection threshold score
		int octaves = 3;  // detection octaves. Use 0 to do single scale
		float patternScale = 1.0f;  // apply this scale to the pattern used for sampling the neighbourhood of a keypoint

        // create BRISK feature detector
		detector = cv::BRISK::create(threshold, octaves, patternScale);
	}
	else if (detectorType.compare("ORB") == 0)
	{
        // ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/db/d95/classcv_1_1ORB.html
        
        // set ORB feature detector parameters
        int nfeatures = 500;  // maximum number of features to retain
		float scaleFactor = 1.2f;  // pyramid decimation ratio, greater than 1 (scaleFactor==2 => classical pyramid)
        int nlevels = 8;  // number of pyramid levels (smallest level has linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel))
		int edgeThreshold = 31;  // size of the border where the features are not detected (should roughly match the patchSize parameter)
		int firstLevel = 0;  // level of pyramid to put source image to (Previous layers are filled with upscaled source image)
		int WTA_K = 2;  // number of points that produce each element of the oriented BRIEF descriptor (default value: 2, other possible values: 3, 4)
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; // default: use HARRIS_SCORE to rank features (or: FAST_SCORE => slightly less stable keypoints, but faster)
		int patchSize = 31;  // size of the patch used by the oriented BRIEF descriptor
		int fastThreshold = 20;  // the fast threshold

        // create ORB feature detector
		detector = cv::ORB::create(
            nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
            WTA_K, scoreType, patchSize, fastThreshold);
	}
	else if (detectorType.compare("KAZE") == 0)
	{
        // KAZE feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/d3/d61/classcv_1_1KAZE.html

        // set KAZE feature detector parameters
        bool extended = false;  // set to enable extraction of extended (128-byte) descriptor
		bool upright = false;  // set to enable use of upright descriptors (non rotation-invariant)
		float threshold = 0.001f; // detector response threshold to accept point
		int nOctaves = 4;  // maximum octave evolution of the image
		int nOctaveLayers = 4;  // default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;  // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER

        // create KAZE feature detector
        detector = cv::KAZE::create(
            extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity); 	
	}
	else if (detectorType.compare("AKAZE") == 0)
	{
        // AKAZE (Accelerated-KAZE) feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/d8/d30/classcv_1_1AKAZE.html

        // set AKAZE feature detector parameters
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;  // options: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT
		int descriptor_size = 0;  // size of the descriptor in bits. 0 -> full size
		int descriptor_channels = 3;  // number of channels in the descriptor (1, 2, 3)
		float threshold = 0.001f;  // detector response threshold to accept point
		int nOctaves = 4;  // maximum octave evolution of the image
		int nOctaveLayers = 4;  // default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;  // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        
        // create AKAZE feature detector
        detector = cv::AKAZE::create(
            descriptor_type, descriptor_size, descriptor_channels, threshold,
            nOctaves, nOctaveLayers, diffusivity);
	}
	else if (detectorType.compare("SIFT") == 0)
	{
        // SIFT (Scale Invariant Feature Transform) feature detector and descriptor extractor
        // https://docs.opencv.org/4.1.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

        // set SIFT feature detector parameters
        int nfeatures = 0;  // number of best features to retain (features are ranked by their scores measured as the local contrast)
        int nOctaveLayers = 3;  // number of layers in each octave (3 is the value used in D. Lowe paper)
        double contrastThreshold = 0.04;  // contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        double edgeThreshold = 10;  // threshold used to filter out edge-like features
        double sigma = 1.6;  // sigma of the Gaussian applied to the input image at the octave #0

        // create SIFT feature detector
		detector = cv::xfeatures2d::SIFT::create(
            nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
	}
    else if (detectorType.compare("SURF") == 0)
    {
        // SURF (Speeded-up robust features) feature detector and descriptor extractor
        // OpenCV: https://docs.opencv.org/4.1.2/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html#details

        // set SURF feature detector parameters
        double hessianThreshold = 100;  // threshold for hessian keypoint detector used in SURF
		int nOctaves = 4;  // number of pyramid octaves the keypoint detector will use
		int nOctaveLayers = 3;  // number of octave layers within each octave
		bool extended = false;  // extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors)
		bool upright = false;  // up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation)

        // create SURF feature detector
        detector = cv::xfeatures2d::SURF::create(
            hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    }
	else {
        // throw error message
        throw "Error: Wrong input argument to detKeypoints(): Detector type not defined!";
	}

    // detect keypoints (timed process)
	double t = (double)cv::getTickCount();  // trigger timer
	detector->detect(img, keypoints); // Detect keypoints
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();  // stop timer
    cout << detectorType
        << " feature detection with n="
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

	// visualize results
	if (bVis)
	{
        // plot image with keypoints
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = detectorType.append(" Detector Results");
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);

        // wait for user key press
		cv::waitKey(0);

        // clear temporary images and variables
        visImage.release();
        windowName.clear();
	}

    // return processing time for keypoint detection
    return t;

}


// compare the strength of the detector response of tow different keypoints for sorting
bool compareKeypointResponse(const cv::KeyPoint & kpt1, const cv::KeyPoint & kpt2)
{
    // return true if response of kpt1 is greater than the response of kpt2, or false otherwise
    return kpt1.response > kpt2.response;
}


// export evaluation results to csv file
void exportResultsToCSV(const std::string fullFilename, boost::circular_buffer<EvalResults> & resultBuffer)
{

    // export evaluation results to a CSV file with one header line and as many rows as there are images
    // @param: fullFilename - full filepath to the csv file
    // @param: resultBuffer - circular buffer holding the results with as many entries as there are images

    // open csv file
    ofstream csv_file;
    csv_file.open(fullFilename, ios::out);

    // write file header using the EvalResults data structure
    csv_file << "imgFilename" << ","
            << "detectorType" << ","
            << "numKeypoints" << ","
            << "t_detKeypoints [s]" << ","
            << "bFocusOnVehicle" << ","
            << "numKeypointsInROI" << ","
            << "bLimitKpts" << ","
            << "numKeypointsInROILimited" << ","
            << "meanDetectorResponse" << ","
            << "meanKeypointDiam" << ","
            << "varianceKeypointDiam_av" << ","
            << "descExtractorType" << ","
            << "t_descKeypoints [s]" << ","
            << "t_detKeypoints + t_descKeypoints [s]" << ","
            << "matcherType" << ","
            << "descriptorType" << ","
            << "selectorType" << ","
            << "numDescMatches" << ","
            << "t_matchDescriptors [s]" << ","
            << "t_detKeypoints + t_descKeypoints + t_matchDescriptors [s]"
            << endl;
    
    // initialize cumulated sums of keypoints / descriptors / descriptor matches over all images
    int numKeypoints_cumulated = 0;
    int numKeypointsInROI_cumulated = 0;
    int numKeypointsInROILimited_cumulated = 0;
    int numDescMatches_cumulated = 0;

    // initialize average of mean keypoint detector response over all images
    double meanDetectorResponse_avg = 0.0;

    // initialize average of mean and variance of the keypoint diameter distributions over all images
    double meanKeypointDiam_avg = 0.0;
    double varianceKeypointDiam_avg = 0.0;

    // initialize average processing times over all images
    double t_detKeypoints_avg = 0.0;
    double t_descKeypoints_avg = 0.0;
    double t_sum_det_desc_avg = 0.0;
    double t_matchDescriptors_avg = 0.0;
    double t_sum_det_desc_match_avg = 0.0;

    // counter
    int cnt = 0;

    // loop over the evaluation results for each image / image pair in the result buffer
    for (auto results = resultBuffer.begin(); results != resultBuffer.end(); results++)
    {
        // write the evaluation results to csv file
        csv_file << results->imgFilename << ","
                << results->detectorType << ","
                << results->numKeypoints << ","
                << results->t_detKeypoints << ","
                << results->bFocusOnVehicle << ","
                << results->numKeypointsInROI << ","
                << results->bLimitKpts << ","
                << results->numKeypointsInROILimited << ","
                << results->meanDetectorResponse << ","
                << results->meanKeypointDiam << ","
                << results->varianceKeypointDiam << ","
                << results->descExtractorType << ","
                << results->t_descKeypoints << ","
                << results->t_sum_det_desc << ","
                << results->matcherType << ","
                << results->descriptorType << ","
                << results->selectorType << ","
                << results->numDescMatches << ","
                << results->t_matchDescriptors << ","
                << results->t_sum_det_desc_match << ","
                << endl;
        
        // cumulate keypoints / descriptors / descriptor matches over all images
        numKeypoints_cumulated += results->numKeypoints;
        numKeypointsInROI_cumulated += results->numKeypointsInROI;
        numKeypointsInROILimited_cumulated += results->numKeypointsInROILimited;
        numDescMatches_cumulated += results->numDescMatches;

        // cumulate mean keypoint detector response over all images
        meanDetectorResponse_avg += results->meanDetectorResponse;

        // cumulate mean and variance of the per image keypoint diameter distribution
        meanKeypointDiam_avg += results->meanKeypointDiam;
        varianceKeypointDiam_avg += results->varianceKeypointDiam;

        // cumulate processing times
        t_detKeypoints_avg += results->t_detKeypoints;
        t_descKeypoints_avg += results->t_descKeypoints;
        t_sum_det_desc_avg += results->t_sum_det_desc;
        t_matchDescriptors_avg += results->t_matchDescriptors;
        t_sum_det_desc_match_avg += results->t_sum_det_desc_match;

        // increment counter
        cnt++;

    }

    // calculate average mean value of detector response over all images
    meanDetectorResponse_avg /= cnt;

    // calculate average keypoint environment (mean and variance) over all images
    meanKeypointDiam_avg /= cnt;
    varianceKeypointDiam_avg /= cnt;

    // calculate average processing times over all iamges
    t_detKeypoints_avg /= cnt;
    t_descKeypoints_avg /= cnt;
    t_sum_det_desc_avg /= cnt;
    t_matchDescriptors_avg /= cnt;
    t_sum_det_desc_match_avg /= cnt;

    // write the cumulated sums of detected keypoints over all images to csv file
    csv_file << "cumulated sum" << ","
                << "" << ","
                << numKeypoints_cumulated << ","
                << "" << ","
                << "" << ","
                << numKeypointsInROI_cumulated << ","
                << "" << ","
                << numKeypointsInROILimited_cumulated << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << numDescMatches_cumulated << ","
                << "" << ","
                << "" << ","
                << endl;

    // write the average processing times over all images to csv file
    csv_file << "average values" << ","
                << "" << ","
                << "" << ","
                << t_detKeypoints_avg << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << meanDetectorResponse_avg << ","
                << meanKeypointDiam_avg << ","
                << varianceKeypointDiam_avg << ","
                << "" << ","
                << t_descKeypoints_avg << ","
                << t_sum_det_desc_avg << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << "" << ","
                << t_matchDescriptors_avg << ","
                << t_sum_det_desc_match_avg << ","
                << endl;

    // close csv file
    csv_file.close();
    
    // print file location where the results have been stored
    cout << "Results have been exported to " << fullFilename << endl;

}


// export overall evaluation results to csv file
void exportOverallResultsToCSV(const std::string fullFilename, std::vector<boost::circular_buffer<EvalResults>> & evalResultBuffers)
{

    // export overall evaluation results on all keypoint detector / descriptor extractor combinations to a CSV file with one 
    // header line and as many rows as there are detector - descriptor extractor combinations
    // @param: fullFilename - full filepath to the csv file
    // @param: resultBuffers - vector of circular buffers holding the evaluation results for each detector / descriptor combination

    // open csv file
    ofstream csv_file;
    csv_file.open(fullFilename, ios::out);

    // write file header using the EvalResults data structure
    csv_file << "id" << ","
            << "keypoint detector" << ","
            << "descriptor extractor" << ","
            << "matcher type" << ","
            << "descriptor type" << ","
            << "selector type" << ","
            << "cumulated sum of keypoints" << ","
            << "cumulated sum of keypoints in ROI" << ","
            << "cumulated sum of keypoints in ROI (limited)" << ","
            << "cumulated sum of matched keypoints in ROI" << ","
            << "t_detKeypoints_avg - average time for keypoint detection (all keypoints) in [s]" << ","
            << "t_descKeypoints_avg - average time for descriptor extraction (only ROI) in [s]" << ","
            << "t_detKeypoints_avg + t_descKeypoints_avg [s]" << ","
            << "t_matchDescriptors_avg - average time for descriptor matching (only ROI) in [s[" << ","
            << "t_detKeypoints_avg + t_descKeypoints_avg + t_matchDescriptors_avg in [s]" << ","
            << endl;

    // initialize detector / descriptor combination id
    int id = 1;

    // loop over the vector of result buffers for each detector / descriptor combination
    for (auto resBuf = evalResultBuffers.begin(); resBuf != evalResultBuffers.end(); resBuf++)
    {

        // initialize cumulated sums of keypoints / descriptors / descriptor matches over all images
        int numKeypoints_cumulated = 0;
        int numKeypointsInROI_cumulated = 0;
        int numKeypointsInROILimited_cumulated = 0;
        int numDescMatches_cumulated = 0;

        // initialize average processing times over all images
        double t_detKeypoints_avg = 0.0;
        double t_descKeypoints_avg = 0.0;
        double t_sum_det_desc_avg = 0.0;
        double t_matchDescriptors_avg = 0.0;
        double t_sum_det_desc_match_avg = 0.0;
        
        // counter
        int cnt = 0;

        // loop over the evaluation results for each image / image pair in the current result buffer
        for (auto results = (*resBuf).begin(); results != (*resBuf).end(); results++)
        {

            // cumulate keypoints / descriptors / descriptor matches over all images
            numKeypoints_cumulated += results->numKeypoints;
            numKeypointsInROI_cumulated += results->numKeypointsInROI;
            numKeypointsInROILimited_cumulated += results->numKeypointsInROILimited;
            if (results->numDescMatches >= 0)
            {
                numDescMatches_cumulated += results->numDescMatches;
            }            

            // cumulate processing times
            t_detKeypoints_avg += results->t_detKeypoints;
            t_descKeypoints_avg += results->t_descKeypoints;
            t_sum_det_desc_avg += results->t_sum_det_desc;
            t_matchDescriptors_avg += results->t_matchDescriptors;
            t_sum_det_desc_match_avg += results->t_sum_det_desc_match;

            // increment counter
            cnt++;

        }

        // calculate average processing times over all iamges
        t_detKeypoints_avg /= cnt;
        t_descKeypoints_avg /= cnt;
        t_sum_det_desc_avg /= cnt;
        t_matchDescriptors_avg /= cnt;
        t_sum_det_desc_match_avg /= cnt;

        // write the cumulated sums of detected keypoints over all images to csv file
        csv_file << id << ","
                << ((*resBuf).end() - 1)->detectorType << ","
                << ((*resBuf).end() - 1)->descExtractorType << ","
                << ((*resBuf).end() - 1)->matcherType << ","
                << ((*resBuf).end() - 1)->descriptorType << ","
                << ((*resBuf).end() - 1)->selectorType << ","
                << numKeypoints_cumulated << ","
                << numKeypointsInROI_cumulated << ","
                << numKeypointsInROILimited_cumulated << ","
                << numDescMatches_cumulated << ","
                << t_detKeypoints_avg << ","
                << t_descKeypoints_avg << ","
                << t_sum_det_desc_avg << ","
                << t_matchDescriptors_avg << ","
                << t_sum_det_desc_match_avg << ","
                << endl;
        
        // increment detector / descriptor combination id
        ++id;

    }

    // close csv file
    csv_file.close();
    
    // print file location where the results have been stored
    cout << "Overall results have been exported to " << fullFilename << endl;

}