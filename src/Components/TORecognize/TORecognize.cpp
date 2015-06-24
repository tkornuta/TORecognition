/*!
 * \file
 * \brief
 * \author Tomek Kornuta,,,
 */

#include <memory>
#include <string>

#include "TORecognize.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

namespace Processors {
namespace TORecognize {

TORecognize::TORecognize(const std::string & name) :
	Base::Component(name),
	prop_filename("filename", std::string("")),
	prop_read_on_init("read_on_init", true),
	prop_matcher_type("descriptor_matcher_type", 0),
	prop_extractor_type("descriptor_extractor_type", 0)
{
	// Register property.
	registerProperty(prop_filename);
	registerProperty(prop_read_on_init);
	registerProperty(prop_matcher_type);
	registerProperty(prop_extractor_type);
}

TORecognize::~TORecognize() {
}

void TORecognize::prepareInterface() {
	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_img", &out_img);

	// Register handlers with their dependencies.
	registerHandler("onNewImage", boost::bind(&TORecognize::onNewImage, this));
	addDependency("onNewImage", &in_img);

	// Register handler - load model manually.
	registerHandler("Load model", boost::bind(&TORecognize::onLoadModelButtonPressed, this));

}

bool TORecognize::onInit() {

	// Initialize extractor.
	current_extractor_type = -1;
	setDescriptorExtractor();

	// Initialize matcher.
	current_matcher_type = -1;
	setDescriptorMatcher();

	if (prop_read_on_init)
		load_model_flag = true;
	else 
		load_model_flag = false;

	return true;
}

bool TORecognize::onFinish() {
	return true;
}

bool TORecognize::onStop() {
	return true;
}

bool TORecognize::onStart() {
	return true;
}

void TORecognize::setDescriptorExtractor(){
	CLOG(LDEBUG) << "setDescriptorExtractor";
	// Check current extractor type.
	if (current_extractor_type == prop_extractor_type) 
		return;

	// Set matcher.
	switch(prop_extractor_type) {
		case 1:
			extractor = DescriptorExtractor::create("SURF");
			CLOG(LNOTICE) << "Using SURF descriptor";
			break;
		case 2:
			extractor = DescriptorExtractor::create("BRIEF");
			CLOG(LNOTICE) << "Using BRIEF descriptor";
			break;
		case 3:
			extractor = DescriptorExtractor::create("BRISK");
			CLOG(LNOTICE) << "Using BRISK descriptor";
			break;
		case 4:
			extractor = DescriptorExtractor::create("ORB");
			CLOG(LNOTICE) << "Using ORB descriptor";
			break;
		case 5:
			extractor = DescriptorExtractor::create("FREAK");
			CLOG(LNOTICE) << "Using FREAK descriptor";
			break;
		case 0 :
		default:
			extractor = DescriptorExtractor::create("SIFT");
			CLOG(LNOTICE) << "Using SIFT descriptor";
			break;
	}//: switch
	// Remember current extractor type.
	current_extractor_type = prop_extractor_type;

	// Reload the model.
	load_model_flag = true;
}


void TORecognize::setDescriptorMatcher(){
	CLOG(LDEBUG) << "setDescriptorMatcher";
	// Check current matcher type.
	if (current_matcher_type == prop_matcher_type) 
		return;

	// Set matcher.
	switch(prop_matcher_type) {
		case 1: matcher = new cv::BFMatcher(NORM_L2, true);
			CLOG(LNOTICE) << "Using BFMatcher with L2 norm and crosscheck";
			break;
		case 2:	matcher = new cv::BFMatcher(NORM_HAMMING);
			CLOG(LNOTICE) << "Using BFMatcher with Hamming norm";
			break;
		case 3:	matcher = new cv::BFMatcher(NORM_HAMMING, true);
			CLOG(LNOTICE) << "Using BFMatcher with Hamming norm and with crosscheck";
			break;
		case 4: matcher = new cv::FlannBasedMatcher();
			CLOG(LNOTICE) << "Using FLANN-based matcher with L2 norm";
			break;
		case 5: matcher = new cv::FlannBasedMatcher(new flann::LshIndexParams(20,10,2));
			CLOG(LNOTICE) << "Using FLANN-based matcher with LSH (Locality-sensitive hashing) norm";
			break;
		case 0 :
		default:matcher = new cv::BFMatcher();
			CLOG(LNOTICE) << "Using BFMatcher with L2 norm";
			break;
	}//: switch
	// Remember current matcher type.
	current_matcher_type = prop_matcher_type;

}


void TORecognize::onLoadModelButtonPressed(){
	CLOG(LDEBUG) << "onLoadModelButtonPressed";
	load_model_flag = true;
}


void TORecognize::loadModel(){
	CLOG(LDEBUG) << "loadModel";

	if (!load_model_flag)
		return;
	load_model_flag = false;

	// Clear the keypoints - just in case...
	model_keypoints.clear();

	// Load the "database" of models - of size 1. ;)
	if ( loadImage(prop_filename, model_img) )
		extractFeatures(model_img, model_keypoints, model_descriptors);

}


bool TORecognize::loadImage(const std::string filename_, cv::Mat & image_) {
	CLOG(LTRACE) << "loadImage";
	try {
	        image_ = imread( filename_ );
		return true;
	} catch (...) {
		CLOG(LWARNING) << "Could not load image from file " << filename_;
		return false;
	}
}


bool TORecognize::extractFeatures(const cv::Mat image_, std::vector<KeyPoint> & keypoints_, cv::Mat & descriptors_) {
	CLOG(LTRACE) << "extractFeatures";
        cv::Mat gray_img;

	// Clear vector of keypoints - just in case...
	keypoints_.clear();

	try {
		// Transform to grayscale - if requred.
		if (image_.channels() == 1)
			gray_img = image_;
		else 
			cvtColor(image_, gray_img, COLOR_BGR2GRAY);

		// Detect the keypoints.
		detector.detect( gray_img, keypoints_ );

		// Extract descriptors (feature vectors).
		extractor->compute( gray_img, keypoints_, descriptors_ );
		return true;
	} catch (...) {
		CLOG(LWARNING) << "Could not extract features from image";
		return false;
	}//: catch
}


void TORecognize::onNewImage()
{
	CLOG(LTRACE) << "onNewImage";
	try {
		// Change descriptor extractor type (if required).
		setDescriptorExtractor();

		// Re-load the model - extract features from model.
		loadModel();

		// Check model.
		if (model_keypoints.size() == 0) {
			CLOG(LWARNING) << "Model not valid. Please load model that contain texture";
			return;
		}//: if

		std::vector<KeyPoint> scene_keypoints;
		cv::Mat scene_descriptors;
		std::vector< DMatch > matches;


		// Input: a colour image.
		cv::Mat scene_img = in_img.read();


		// Extract features from scene.
		extractFeatures(scene_img, scene_keypoints, scene_descriptors);

		CLOG(LNOTICE) << "Model features: " << model_keypoints.size();
		CLOG(LNOTICE) << "Scene features: " << scene_keypoints.size();

		// Change matcher type (if required).
		setDescriptorMatcher();

		// Find matches.
		matcher->match( model_descriptors, scene_descriptors, matches );

		CLOG(LNOTICE) << "Matches found: " << matches.size();

		// Draw all matches.
		Mat img_matches1;
		drawMatches( model_img, model_keypoints, scene_img, scene_keypoints,
		             matches, img_matches1, Scalar::all(-1), Scalar::all(-1),
		             vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

//		out_img.write(img_matches1);


		// Filtering.
		double max_dist = 0;
		double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < matches.size(); i++ ) {
			double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}//: for

		CLOG(LDEBUG) << "Max dist : " << max_dist;
		CLOG(LDEBUG) << "Min dist : " << min_dist;

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		std::vector< DMatch > good_matches;

		for( int i = 0; i < matches.size(); i++ ) {
			if( matches[i].distance < 3*min_dist )
				good_matches.push_back( matches[i]);
		}//: for

		CLOG(LNOTICE) << "Good matches: " << good_matches.size();

		Mat img_matches2;
		drawMatches( model_img, model_keypoints, scene_img, scene_keypoints,
		             good_matches, img_matches2, Scalar::all(-1), Scalar::all(-1),
		             vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		out_img.write(img_matches2);

		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		// Get the keypoints from the good matches.
		for( int i = 0; i < good_matches.size(); i++ ) {
		  obj.push_back( model_keypoints [ good_matches[i].queryIdx ].pt );
		  scene.push_back( scene_keypoints [ good_matches[i].trainIdx ].pt );
		}//: for

		// Find homography between corresponding points.
		Mat H = findHomography( obj, scene, CV_RANSAC );

		// Get the corners from the detected "object model".
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0,0);
		obj_corners[1] = cvPoint( model_img.cols, 0 );
		obj_corners[2] = cvPoint( model_img.cols, model_img.rows );
		obj_corners[3] = cvPoint( 0, model_img.rows );
		std::vector<Point2f> scene_corners(4);

		// Transform corners with found homography.
		perspectiveTransform( obj_corners, scene_corners, H);

		//Draw lines between the corners on the input image.
		//Mat img_matches3 = img_matches2;

		line( img_matches2, scene_corners[0] + Point2f( scene_img.cols, 0), scene_corners[1] + Point2f( scene_img.cols, 0), Scalar(0, 255, 0), 4 );
		line( img_matches2, scene_corners[1] + Point2f( scene_img.cols, 0), scene_corners[2] + Point2f( scene_img.cols, 0), Scalar( 0, 255, 0), 4 );
		line( img_matches2, scene_corners[2] + Point2f( scene_img.cols, 0), scene_corners[3] + Point2f( scene_img.cols, 0), Scalar( 0, 255, 0), 4 );
		line( img_matches2, scene_corners[3] + Point2f( scene_img.cols, 0), scene_corners[0] + Point2f( scene_img.cols, 0), Scalar( 0, 255, 0), 4 );

//		out_img.write(img_matches2);

	} catch (...) {
		CLOG(LERROR) << "onNewImage failed\n";
	}
}


} //: namespace TORecognize
} //: namespace Processors
