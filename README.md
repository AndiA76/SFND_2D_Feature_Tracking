# SFND 2D Feature Tracking

<p style="color:black;font-size:14px;">
<img src="images/SIFT_keypoints.png" width="820" height="248" />
<em><br>Example: Results of SIFT keypoint detection using default parameters (1438 keypoints detected in image 0000.png in 92.523 ms)</em>
</p>
  
The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:
  
* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 
  
See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* Boost C++ libraries >= 1.65.1 for circular data frame buffer implementation
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Install requirements.
2. Build OpenCV from source enabling the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors and install.
   * Howto's for OpenCV 4.1.0 and 4.1.2:
     * [How to install OpenCV 4.1.0 with CUDA 10.0 in Ubuntu distro 18.04](https://gist.github.com/DSamuylov/ebae5d4dd4e2ba6bd4af32b44cf97b98)
     * [How to install OpenCV 4.2.0 with CUDA 10.0 in Ubuntu distro 18.04](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
   * Remark: CUDA is not required here.
3. Clone this repo.
4. Make a build directory in the top level directory: `mkdir build && cd build`
5. Compile: `cmake .. && make`
6. Run it: `./2D_feature_tracking`.

# Mid-Term Project Submission

## Data Buffer

### MP.1 Data Buffer Optimization

_Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end._

#### Circular Data Buffer Implementation using Boost

As we store structued data frame objects of struct DataFame (s. [dataStructures.h](/src/dataStructures.h)) a generic ringbuffer class is needed. Boost C++ Libraries offers a generic class template of a circular ringbuffer, which meets all requirements. Because boost c++ libraries are well tested and only a one line change is needed, I have deciced to choose this approach of [Boost.Circular_Buffer](https://www.boost.org/doc/libs/1_65_1/doc/html/circular_buffer.html) implementation.  
  
Check installed boost version
```
cout << "Using Boost version "
	<< BOOST_VERSION / 100000     << "."  // major version
	<< BOOST_VERSION / 100 % 1000 << "."  // minor version
	<< BOOST_VERSION % 100                // patch level
	<< endl;
```
  
Replace the following initialization in [MidTermProject_Camera_Student.cpp](/src/MidTermProject_Camera_Student.cpp):
```
vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time (constantly growing!)
```
with:
```
boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize); // create circular buffer for DataFrame structures
```
  
Print current buffer size and maximum capacity of the circular data buffer
```
cout << "Circular data buffer capacity in use = "
	<< dataBuffer.size()
	<< " of max. "
	<< dataBuffer.capacity()
	<< " data frames."
	<< endl;
```

## Keypoints / Features

### MP.2 Keypoint / Feature Detection

The following keypoint resp. feature detectors from OpenCV have been implemented:
- SHI-TOMASI (given already by Udacity code skeleton)
- HARRIS
- FAST
- BRISK
- ORB
- KAZE
- AKAZE
- SIFT (non-free!!!)
- SURF (non-free!!!)

#### Implementation of the Harris Corner Detector

<p style="color:black;font-size:14px;">
<img src="images/HARRIS_keypoints.png" width="820" height="248" />
<em><br>Example: Results of HARRIS keypoint detection using default parameters (115 keypoints detected in image 0000.png in 15.845 ms)</em>
</p>
  
Excerpt from [MidTermProject_Camera_Student.cpp](/src/MidTermProject_Camera_Student.cpp):
```
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
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    // return processing time for keypoint detection
    return t;

}
```

#### Impementation of the other Feature Detectors (FAST, BRISK, ...)

<p style="color:black;font-size:14px;">
<img src="images/AKAZE_keypoints.png" width="820" height="248" />
<em><br>Example: Results of AKAZE keypoint detection using default parameters (1351 keypoints detected in image 0000.png in 59.583 ms)</em>
</p>
  
Excerpt from [MidTermProject_Camera_Student.cpp](/src/MidTermProject_Camera_Student.cpp):
```
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

    cout << "Selected detector type = " << detectorType << endl;


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
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = detectorType.append(" Detector Results");
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}

    // return processing time for keypoint detection
    return t;

}
```

### Keypoint Filtering

#### Keypoint filtering using Target Bounding Boxes

In a potential future application of this code bounding boxes would be provided by an additional object detection CNN (e. g. bounding boxes for detected vehicles). Here a given bounding box for the target vehicle (obstacle) is used as a placeholder to filter out keypoints within this region of interest. In a later application this fixed bounding box needs to be replaced by the output of an appropriate object detector.
  
<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_AKAZE_keypoints.png" width="820" height="496" />
<em><br>Example: AKAZE keypoints before and after region-of-interest filtering using a given target bounding box</em>
</p>
  
Excerpt from [MidTermProject_Camera_Student.cpp](/src/MidTermProject_Camera_Student.cpp):
```
//// STUDENT ASSIGNMENT
//// TASK MP.3 -> only keep keypoints on the preceding vehicle

// only keep keypoints on the preceding vehicle
// remark: This is a temporary solution that will be replace with bounding boxes provided by an object detection CNN!
bool bFocusOnVehicle = true;  // true: only consider keypoints within the bounding box; false: consider all keypoints
results.bFocusOnVehicle = bFocusOnVehicle;  // store bFocusOnVehicle flag in results
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
```

#### Optional: Limiting the Maximum Number of Keypoints

For debugging purpose and to get a better overview of the selected keypoints an optional filter is included after applying the target bounding box selection filter here that limits the maximum number of keypoints to be considered. As keypoint / feature detectors do not necessarily sort the detected keypoints, the keypoints need to be sorted by the strength of the feature detector response before limiting their number. Otherwise, keypoints with lower strength would be considered in the final selection prior to much better keypoints with a stronger feature detector response. After sorting the first maxKeypoints are kept in the list, the rest is deleted. 
  
<p style="color:black;font-size:14px;">
<img src="images/limit_number_of_AKAZE_keypoints_in_target_bounding_box.png" width="820" height="496" />
<em><br>Example: AKAZE keypoints located within a given target bounding box before and after limiting their maximum number to 50 (in this example)</em>
</p>
  
Excerpt from [matching2D.hpp](/src/matching2D.hpp):
```
bool compareKeypointResponse(const cv::KeyPoint & p1, const cv::KeyPoint & p2);
```
  
Excerpt from [matching2D_Student.cpp](/src/matching2D_Student.cpp):
```
// compare the strength of the detector response of tow different keypoints for sorting
bool compareKeypointResponse(const cv::KeyPoint & kpt1, const cv::KeyPoint & kpt2)
{
    // return true if response of kpt1 is greater than the response of kpt2, or false otherwise
    return kpt1.response > kpt2.response;
}
```
  
Excerpt from [MidTermProject_Camera_Student.cpp](/src/MidTermProject_Camera_Student.cpp):
```
// sort keypoints according to the strength of the detector response (not every detector sorts the keypoints automatically!)
sort(keypoints.begin(), keypoints.end(), compareKeypointResponse);

// keep the first maxKeypoints from the list sorted by descending order of the detector response
keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
```

## Keypoint / Feature Descriptors

### MP.4 Extraction of Keypoint / Feature Descriptors

The following keypoint resp. feature descriptors from OpenCV have been implemented:
- BRISK (given by Udacity code skeleton)
- BRIEF
- ORB
- FREAK
- KAZE (only works with KAZE / AKAZE keypoints)
- AKAZE (only works with KAZE / AKAZE keypoints)
- SIFT (non-free!!!)
- SURF (non-free!!!)

#### Impementation of the other Feature Descriptors (BRIEF, ORB, ...)

Excerpt from [matching2D_Student.cpp](/src/matching2D_Student.cpp):
```
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

    cout << "Selected feature descriptor (extractor) type = " << descExtractorType << endl;

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
```

### MP.5 and MP.6 Descriptor Matching

#### Adding FLANN-based Keypoints Matching and K-Nearest Neighbor Selection with Descriptor Distance Ratio Test

An implementation of brute force keypoint matching with an nearest neighbor search strategy to find the best match was already given by the code skeleton provided by Udacity. A FLANN-based keypoints matcher using KD-tree approach with k-nearest neighbor search to find the best match has been added to the code (s. next section). Some sample results for SIFT keypoints are shonw in the image.
  
Excerpt from [matching2D_Student.cpp](/src/matching2D_Student.cpp):
```
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
        cout << " KNN matching with n = " << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

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
```

#### MP.5 FLANN-based Keypoint Matching

_TASK MP.5 -> Add FLANN matching in file matching2D.cpp as a more time efficient method compared to brute-force matching._
  
<p style="color:black;font-size:14px;">
<img src="images/FLANN-based_NN_matching_of_SIFT-BRISK-keypoints.png" width="820" height="248" />
<em><br>Example: FLANN-based matching of SIFT keypoints (with BRISK descriptors) located within a given target bounding box using nearest neighbor selection</em>
</p>
  
Above image shows two subsequent image frames plotted besides one another. The left side of the image shows the image frame captured first, the right side of the image shows the sub-sequent image frame. The keypoints found within the rectangular target bounding box (using SIFT in this example) have been marked. For each of those keypoints a descriptor has been calculated (using BRISK in this example).
Afterwards, FLANN-based matching with nearest neighbor selection has been applied to match the keypoint descriptors in the left image with similar ones in the right image. This may also lead to some false positive matches.
False postive matches can be recognized by non-horizontal connection lines between keypoints in the left and the right image, for instance. Such false positives are easy to spot for a human eye. However, some other false positive matches may in fact have horizontal connection lines, but they do not end at the right spot. Such false positives are more difficult to recognize at one glance.

#### MP.6 Descriptor Distance Ratio Test als False Positive Filter

_TASK MP.6 -> Add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp in order to reduce the number of false postive matches._
  
<p style="color:black;font-size:14px;">  
<img src="images/FLANN-based_KNN_matching_of_SIFT-BRISK-keypoints.png" width="820" height="248" />
<em><br>Example: FLANN-based matching of SIFT keypoints (with BRISK descriptors) located within a given target bounding box using k-nearest neighbor selection (k = 2) and distance ratio test (minDescDistRatio = 0.8)</em>
<em><br>
  
Above image shows two subsequent image frames plotted besides one another. The left side of the image shows the image frame captured first, the right side of the image shows the sub-sequent image frame. The keypoints found within the rectangular target bounding box (using SIFT in this example) have been marked. For each of those keypoints a descriptor has been calculated (using BRISK in this example).
Afterwards, FLANN-based matching has been applied to match the keypoint descriptorss in the left image with similar keypoint descriptors in the right image. K-nearest neightbor (k=2) search has then been applied to select the k best choices. This allows two possible options in this case. 
A subsequent filtering step based on a descriptor distance ratio test with a threshold of 0.8 has been used afterwards to reduce the number of false positive matches.  
  
When comparing the results of FLANN-based keypoints matching with k-nearest neighbor search and descriptor distance ratio filtering (s. image above) with FLANN-based keypoint matching and nearest neigbor search (s. image from the previous section) one realizes that the descriptor distance ratio test effectively reduces false positive matches. In this example, it removes all false positive matches, but this is not necessarily the case in general.

## Performance

### Configuration

#### Allowed Settings for Keypoint Detector, Keypoint Descriptor and Matcher

### MP.7 - Performance Evaluation w. r. t. Total Number of Detected Keypoints

_TASK MP.7 -> For all implemented keypoint detectors: Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size._

<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_SHITOMASI_keypoints.png" width="820" height="496" />
<em><br>Example: SHI-TOMASI keypoints before and after region-of-interest filtering using a given target bounding box for the leading vehicle</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_HARRIS_keypoints.png" width="820" height="496" />
<em><br>Example: HARRIS keypoints before and after region-of-interest filtering using a given target bounding box for the leading vehicle</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_FAST_keypoints.png" width="820" height="496" />
<em><br>Example: FAST keypoints before and after region-of-interest filtering using a given target bounding box for the leading vehicle</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_BRISK_keypoints.png" width="820" height="496" />
<em><br>Example: BRISK keypoints before and after region-of-interest filtering using a given target bounding box for the leading vehicle</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_ORB_keypoints.png" width="820" height="496" />
<em><br>Example: ORB keypoints before and after region-of-interest filtering using a given target bounding box for the leading vehicle</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_AKAZE_keypoints.png" width="820" height="496" />
<em><br>Example: AKAZE keypoints before and after region-of-interest filtering using a given target bounding box for the leading vehicle</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/bounding_box_filtering_SIFT_keypoints.png" width="820" height="496" />
<em><br>Example: SIFT keypoints before and after region-of-interest filtering using a given target bounding box for the leading vehicle</em>
</p>

|image no.|SHI-TOMASI|HARRIS|FAST|BRISK|ORB|AKAZE|SIFT|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0000.png|125|17|419|264|92|166|138|
|0001.png|118|14|429|282|102|157|132|
|0002.png|123|18|404|282|106|161|124|
|0003.png|120|21|423|277|113|155|137|
|0004.png|120|26|386|297|109|163|134|
|0005.png|113|43|414|297|125|164|140|
|0006.png|114|18|418|289|130|173|137|
|0007.png|123|31|406|272|129|175|148|
|0008.png|111|26|396|267|127|177|159|
|0009.png|112|34|401|254|128|179|137|
|cumulated sum|1179|248|4094|2763|1161|1670|1386|  
Table 1: Keypoints on target vehicle for different detector types

### MP.8 - Performance Evaluation w. r. t. Number of Keypoint Matches

_TASK MP.8 -> Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8._

|detector/descriptor|cumulated sum of matched keypoints in ROI|
|:-:|:-:|
|SHI-TOMASI/BRISK|767|
|HARRIS/BRISK|142|
|FAST/BRISK|2183|
|BRISK/BRISK|1570|
|ORB/BRISK|751|
|AKAZE/BRISK|1215|
|SIFT/BRISK|592|
|SHI-TOMASI/BRIEF|944|
|HARRIS/BRIEF|173|
|FAST/BRIEF|2831|
|BRISK/BRIEF|1704|
|ORB/BRIEF|545|
|AKAZE/BRIEF|1266|
|SIFT/BRIEF|702|
|SHI-TOMASI/ORB|908|
|HARRIS/ORB|162|
|FAST/ORB|2768|
|BRISK/ORB|1516|
|ORB/ORB|763|
|AKAZE/ORB|1182|
|SIFT/ORB|n. a.|
|SHI-TOMASI/FREAK|768|
|HARRIS/FREAK|144|
|FAST/FREAK|2233|
|BRISK/FREAK|1524|
|ORB/FREAK|420|
|AKAZE/FREAK|1187|
|SIFT/FREAK|593|
|SHI-TOMASI/AKAZE|n. a.|
|HARRIS/AKAZE|n. a.|
|FAST/AKAZE|n. a.|
|BRISK/AKAZE|n. a.|
|ORB/AKAZE|n. a.|
|AKAZE/AKAZE|1259|
|SIFT/AKAZE|n. a.|
|SHI-TOMASI/SIFT|927|
|HARRIS/SIFT|163|
|FAST/SIFT|2782|
|BRISK/SIFT|1648|
|ORB/SIFT|763|
|AKAZE/SIFT|1270|
|SIFT/SIFT|800|  
Table 2: Sum of matched keypoints over 10 images for different detector - descriptor combinations

Remarks:
* KAZE/AKZE descriptor extractors only work with KAZE/AKAZE keypoints
* SIFT/ORB not available due to memory allocation error (out of memory)
* Use DES_HOG seting and L2 norm when SIFT is used as descriptor extractor
* USe DES_BINARY and Hamming norm for all other descriptor extractors in table 2

### MP.9 - Performance Evaluation w. r. t. Processing Time

_TASK MP.9 -> Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet. Based on this information suggest the TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles, and justify the choice based on the obervations collected during the experiments with the code._ 

|detector/descriptor|average time for keypoint detection (all keypoints) in [s]|average time for descriptor extraction (only ROI) in [s]|sum in [s]|
|:-:|:-:|:-:|:-:|
|SHI-TOMASI/BRISK|0.015512|0.00156241|0.0170744|
|HARRIS/BRISK|0.013947|0.000742011|0.014689|
|FAST/BRISK|0.00219866|0.00367454|0.0058732|
|BRISK/BRISK|0.0365125|0.00257855|0.039091|
|ORB/BRISK|0.0148211|0.00124744|0.0160685|
|AKAZE/BRISK|0.0504828|0.00163582|0.0521186|
|SIFT/BRISK|0.085072|0.00142566|0.0864977|
|SHI-TOMASI/BRIEF|0.0215989|0.00113854|0.0227374|
|HARRIS/BRIEF|0.0220589|0.000471153|0.0225301|
|FAST/BRIEF|0.00405871|0.00294237|0.00700108|
|BRISK/BRIEF|0.0367236|0.000884496|0.0376081|
|ORB/BRIEF|0.0211237|0.00078423|0.0219079|
|AKAZE/BRIEF|0.0583681|0.000646203|0.0590143|
|SIFT/BRIEF|0.0941502|0.000598181|0.0947484|
|SHI-TOMASI/ORB|0.023451|0.00119424|0.0246452|
|HARRIS/ORB|0.0201263|0.00088511|0.0210114|
|FAST/ORB|0.00388872|0.0027849|0.00667362|
|BRISK/ORB|0.0373278|0.00394393|0.0412717|
|ORB/ORB|0.0210424|0.00517446|0.0262169|
|AKAZE/ORB|0.059152|0.00237376|0.0615258|
|SIFT-ORB|n. a.|n. a.|n. a.|
|SHI-TOMASI/FREAK|0.0170216|0.0310281|0.0480498|
|HARRIS/FREAK|0.0204718|0.029495|0.0499667|
|FAST/FREAK|0.00395992|0.0379278|0.0418877|
|BRISK/FREAK|0.0362705|0.0303552|0.0666257|
|ORB/FREAK|0.0221652|0.0329646|0.0551298|
|AKAZE/FREAK|0.0551175|0.0297794|0.0848969|
|SIFT/FREAK|0.0948692|0.0310217|0.125891|
|SHI-TOMASI/AKAZE|n. a.|n. a.|n. a.|
|HARRIS/AKAZE|n. a.|n. a.|n. a.|
|FAST/AKAZE|n. a.|n. a.|n. a.|
|BRISK/AKAZE|n. a.|n. a.|n. a.|
|ORB/AKAZE|n. a.|n. a.|n. a.|
|AKAZE/AKAZE|0.061714|0.0392336|0.100948|
|SIFT/AKAZE|n. a.|n. a.|n. a.|
|SHI-TOMASI/SIFT|0.0331213|0.0180846|0.0512059|
|HARRIS/SIFT|0.0249541|0.0167003|0.0416545|
|FAST/SIFT|0.0048849|0.032225|0.0371099|
|BRISK/SIFT|0.0389692|0.0278033|0.0667725|
|ORB/SIFT|0.0208197|0.0331877|0.0540074|
|AKAZE/SIFT|0.0570165|0.0180464|0.0750629|
|SIFT/SIFT|0.0908357|0.0674513|0.158287|  
Table 3: Average processing time for keypoint detection and descriptor extraction over 10 images for different detector - descriptor combinations

### Final Preferrence
When taking the cumulated number of descriptor matches on the target vehicle and the processing time for keypoint detection and descriptor extraction as a selection criterion the following three detector / descriptor combinations would be my preference:
  
|detector/descriptor|cumulated sum of matched keypoints in ROI|average processing time for keypoint detection and descriptor extraction sum in [s]|
|:-:|:-:|:-:|
|FAST/BRISK|2183|0.0058732|
|FAST/BRIEF|2831|0.00700108|
|FAST/ORB|2768|0.00667362|  
Table 4: Number of descriptor matches and processing time for keypoint detection and descriptor extraction of the preferred variants

Below images show descriptor matching between image 0000.png and 0001.png for using the three detector-descriptor combination from table 4.
We can see that the keypoints, which FAST provides within the bounding box around the target vehicle, are mainly located around the target vehicle's outer countour and for great part also lie directly on the target vehicle. So this result does not contradict to use FAST as a feature detector for this task.
The images below also show that descriptor matching looks ok in all three cases as there seem to be not many false positives (e. g. diagonal connection lines). What concerns the latter issue the combination FAST and BRIEF might yield less false positives than the combination FAST and BRISK, but this is hard to tell in general from just one or a few examples.

<p style="color:black;font-size:14px;">
<img src="images/BF-KNN_matching_of_FAST-BRISK-keypoints.png" width="820" height="248" />
<em><br>Example: BF matching with KNN selection (k=2) of FAST keypoints with BRISK descriptors</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/BF-KNN_matching_of_FAST-BRIEF-keypoints.png" width="820" height="248" />
<em><br>Example: BF matching with KNN selection (k=2) of FAST keypoints with BRIEF descriptors</em>
</p>

<p style="color:black;font-size:14px;">
<img src="images/BF-KNN_matching_of_FAST-ORB-keypoints.png" width="820" height="248" />
<em><br>Example: BF matching with KNN selection (k=2) of FAST keypoints with ORB descriptors</em>
</p>

So the preferred keypoint detector / descriptor combinations would be: __FAST/BRISK__, __FAST/BRIEF__, __FAST/ORB__.

# Reference

[1]: Jianbo Shi and Carlo Tomasi. Good features to track. In Computer Vision and Pattern Recognition, 1994. Proceedings CVPR'94., 1994 IEEE Computer Society Conference on, pages 593–600. IEEE, 1994.  

[2]: Chris Harris and Mike Stephens. A Combined Corner and Edge Detector. In Proceedings of the Alvey Vision Conference 1998, pages 23.1-23.6. Alvey Vision Club, September 1998.  

[3]: Edward Rosten and Tom Drummond. Machine learning for high-speed corner detection. In Computer Vision–ECCV 2006, pages 430–443. Springer, 2006.  

[4]: Stefan Leutenegger, Margarita Chli, and Roland Yves Siegwart. Brisk: Binary robust invariant scalable keypoints. In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 2548–2555. IEEE, 2011.  

[5]: Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua. Brief: Binary robust independent elementary features. In Computer Vision–ECCV 2010, pages 778–792. Springer, 2010.  

[6]: Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: an efficient alternative to sift or surf. In Computer Vision (ICCV), 2011 IEEE International Conference on, pages 2564–2571. IEEE, 2011.  

[7]: Alexandre Alahi, Raphael Ortiz, and Pierre Vandergheynst. Freak: Fast retina keypoint. In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on, pages 510–517. Ieee, 2012.  

[8]: Pablo Fernández Alcantarilla, Adrien Bartoli, and Andrew J Davison. Kaze features. In Computer Vision–ECCV 2012, pages 214–227. Springer, 2012.  

[9]: Pablo F Alcantarilla, Jesús Nuevo, and Adrien Bartoli. Fast explicit diffusion for accelerated features in nonlinear scale spaces. Trans. Pattern Anal. Machine Intell, 34(7):1281–1298, 2011.  

[10]: David G Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2):91–110, 2004.  

[11]: Herbert Bay, Tinne Tuytelaars, and Luc Van Gool. Surf: Speeded up robust features. Computer Vision–ECCV 2006, pages 404–417, 2006.  




