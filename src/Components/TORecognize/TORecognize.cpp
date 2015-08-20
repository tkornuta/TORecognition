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
	prop_detector_type("keypoint_detector_type", 0),
	prop_extractor_type("descriptor_extractor_type", 0),
	prop_matcher_type("descriptor_matcher_type", 0),
	prop_returned_model_number("returned_model_number", 0),
	prop_recognized_object_limit("recognized_object_limit", 1)
{
	// Register property.
	registerProperty(prop_filename);
	registerProperty(prop_read_on_init);
	registerProperty(prop_detector_type);
	registerProperty(prop_extractor_type);
	registerProperty(prop_matcher_type);
	registerProperty(prop_returned_model_number);
	registerProperty(prop_recognized_object_limit);
}

TORecognize::~TORecognize() {
}

void TORecognize::prepareInterface() {
	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_img_all_correspondences", &out_img_all_correspondences);
	registerStream("out_img_good_correspondences", &out_img_good_correspondences);
	registerStream("out_img_object", &out_img_object);

	// Register handlers with their dependencies.
	registerHandler("onNewImage", boost::bind(&TORecognize::onNewImage, this));
	addDependency("onNewImage", &in_img);

	// Register handler - load model manually.
	registerHandler("Load model", boost::bind(&TORecognize::onLoadModelButtonPressed, this));

}

bool TORecognize::onInit() {

	// Initialize detector.
	current_detector_type = -1;
	setKeypointDetector();

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


void TORecognize::setKeypointDetector(){
	CLOG(LDEBUG) << "setKeypointDetector";
	// Check current detector type.
	if (current_detector_type == prop_detector_type) 
		return;

	// Set detector.
	switch(prop_detector_type) {
		case 1:
			detector = FeatureDetector::create("STAR");
			CLOG(LNOTICE) << "Using STAR detector";
			break;
		case 2:
			detector = FeatureDetector::create("SIFT");
			CLOG(LNOTICE) << "Using SIFT detector";
			break;
		case 3:
			detector = FeatureDetector::create("SURF");
			CLOG(LNOTICE) << "Using SURF detector";
			break;
		case 4:
			detector = FeatureDetector::create("ORB");
			CLOG(LNOTICE) << "Using ORB detector";
			break;
		case 5:
			detector = FeatureDetector::create("BRISK");
			CLOG(LNOTICE) << "Using BRISK detector";
			break;

		case 6:
			detector = FeatureDetector::create("MSER");
			CLOG(LNOTICE) << "Using MSER detector";
			break;
		case 7:
			detector = FeatureDetector::create("GFTT");
			CLOG(LNOTICE) << "Using GFTT detector";
			break;
		case 8:
			detector = FeatureDetector::create("HARRIS");
			CLOG(LNOTICE) << "Using HARRIS detector";
			break;
		case 9:
			detector = FeatureDetector::create("Dense");
			CLOG(LNOTICE) << "Using Dense detector";
			break;
		case 10:
			detector = FeatureDetector::create("SimpleBlob");
			CLOG(LNOTICE) << "Using SimpleBlob detector";
			break;
		case 0 :
		default:
			detector = FeatureDetector::create("FAST");
			CLOG(LNOTICE) << "Using FAST detector";
			break;
	}//: switch
	// Remember current detector type.
	current_detector_type = prop_detector_type;

	// Reload the model.
	load_model_flag = true;
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



void TORecognize::loadSingleModel(std::string filename_, std::string name_){
	CLOG(LTRACE) << "loadSingleModel";
	cv::Mat model_img;

	if ( loadImage(filename_, model_img) ) {
		std::vector<KeyPoint> model_keypoints;
		cv::Mat model_descriptors;
		extractFeatures(model_img, model_keypoints, model_descriptors);

		// Add to database.
		models_imgs.push_back(model_img);
		models_keypoints.push_back(model_keypoints);
		models_descriptors.push_back(model_descriptors);
		models_names.push_back(name_);
		CLOG(LNOTICE) << "Successfull load of model (" << models_names.size()-1 <<"): "<<models_names[models_names.size()-1];
	}//: if
}

void TORecognize::loadModels(){
	CLOG(LDEBUG) << "loadModels";

	if (!load_model_flag)
		return;
	load_model_flag = false;

	// Clear database.
	models_imgs.clear();
	models_keypoints.clear();
	models_descriptors.clear();
	models_names.clear();

	// Load single model - for now...
//	loadSingleModel(prop_filename, "c3po-ultra-model");

	loadSingleModel("/home/tkornuta/discode_ecovi/DCL/TORecognition/data/dilmah_ceylon_lemon.jpg", "dilmah ceylon lemon");
//	loadSingleModel("/home/tkornuta/discode_ecovi/DCL/TORecognition/data/lipton_earl_grey_classic.jpg", "lipton earl grey classic");
//	loadSingleModel("/home/tkornuta/discode_ecovi/DCL/TORecognition/data/lipton_earl_grey_lemon.jpg", "lipton earl grey lemon");
	loadSingleModel("/home/tkornuta/discode_ecovi/DCL/TORecognition/data/lipton_green_tea_citrus.jpg", "lipton green tea citrus");
	loadSingleModel("/home/tkornuta/discode_ecovi/DCL/TORecognition/data/lipton_tea_lemon.jpg", "lipton tea lemon");
	loadSingleModel("/home/tkornuta/discode_ecovi/DCL/TORecognition/data/twinings_earl_grey.jpg", "twinings earl grey");
	loadSingleModel("/home/tkornuta/discode_ecovi/DCL/TORecognition/data/ahmad_daarjeling.png", "ahmad daarjeling");

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

	try {
		// Transform to grayscale - if requred.
		if (image_.channels() == 1)
			gray_img = image_;
		else 
			cvtColor(image_, gray_img, COLOR_BGR2GRAY);

		// Detect the keypoints.
		detector->detect( gray_img, keypoints_ );

		// Extract descriptors (feature vectors).
		extractor->compute( gray_img, keypoints_, descriptors_ );
		return true;
	} catch (...) {
		CLOG(LWARNING) << "Could not extract features from image";
		return false;
	}//: catch
}


void TORecognize::storeObjectHypothesis(std::string name_, cv::Point2f center_, std::vector<cv::Point2f> corners_, double score_) {
	// Special case: do not insert anything is smaller than one;)
	if (prop_recognized_object_limit<1)
		return;

	// First case: insert first oid.
	if (recognized_names.size() == 0) {
		recognized_names.push_back(name_);
		recognized_centers.push_back(center_);
		recognized_corners.push_back(corners_);
		recognized_scores.push_back(score_);
		return;
	}//: if

	// Iterators.
	std::vector<std::string>::iterator names_it = recognized_names.begin();
	std::vector<cv::Point2f>::iterator centers_it = recognized_centers.begin();
	std::vector<std::vector<cv::Point2f> >::iterator corners_it= recognized_corners.begin();
	std::vector<double>::iterator scores_it= recognized_scores.begin();

	// Second case: insert in proper order.
	bool added = false;
	for (; names_it<recognized_names.end(); names_it++, centers_it++, corners_it++, scores_it++) {
		if (*scores_it < score_){
			// Insert here! (i.e. before)
			recognized_names.insert(names_it, name_);
			recognized_centers.insert(centers_it, center_);
			recognized_corners.insert(corners_it, corners_);
			recognized_scores.insert(scores_it, score_);
			added = true;
			break;
		}//: if
	}//: for*/

	// Third case: insert at the end.
	if (!added){
		recognized_names.push_back(name_);
		recognized_centers.push_back(center_);
		recognized_corners.push_back(corners_);
		recognized_scores.push_back(score_);
	}//: if

	// Limit the size of vectors.
	if (recognized_names.size() > prop_recognized_object_limit){
		recognized_names.pop_back();
		recognized_centers.pop_back();
		recognized_corners.pop_back();
		recognized_scores.pop_back();
	}//: if
}


void TORecognize::onNewImage()
{
	CLOG(LTRACE) << "onNewImage";
	try {
		// Change keypoint detector and descriptor extractor types (if required).
		setKeypointDetector();
		setDescriptorExtractor();

		// Re-load the model - extract features from model.
		loadModels();

		std::vector<KeyPoint> scene_keypoints;
		cv::Mat scene_descriptors;
		std::vector< DMatch > matches;

		// Clear vectors! ;)
		recognized_names.clear();
		recognized_centers.clear();
		recognized_corners.clear();
		recognized_scores.clear();


		// Load image containing the scene.
		cv::Mat scene_img = in_img.read();



		// Extract features from scene.
		extractFeatures(scene_img, scene_keypoints, scene_descriptors);
		CLOG(LINFO) << "Scene features: " << scene_keypoints.size();

		// Iterate - try to detect each model one by one.
		for (unsigned int m=0; m < models_imgs.size(); m++) {
			CLOG(LDEBUG) << "Trying to recognize model (" << m <<"): " << models_names[m];
	
			if ((models_keypoints[m]).size() == 0) {
				CLOG(LWARNING) << "Model "<< models_names[m] << " not valid as it does not contain texture";
				continue;
			}//: if

			CLOG(LDEBUG) << "Model features: " << models_keypoints[m].size();

			// Change matcher type (if required).
			setDescriptorMatcher();

			// Find matches.
			matcher->match( models_descriptors[m], scene_descriptors, matches );

			CLOG(LDEBUG) << "Matches found: " << matches.size();

			if (m == prop_returned_model_number) {
				// Draw all found matches.
				Mat img_matches1;
				drawMatches( models_imgs[m], models_keypoints[m], scene_img, scene_keypoints,
					     matches, img_matches1, Scalar::all(-1), Scalar::all(-1),
					     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
				out_img_all_correspondences.write(img_matches1);
			}//: if


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

			CLOG(LDEBUG) << "Good matches: " << good_matches.size();

			// Localize the object
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;

			// Get the keypoints from the good matches.
			for( int i = 0; i < good_matches.size(); i++ ) {
			  obj.push_back( models_keypoints [m] [ good_matches[i].queryIdx ].pt );
			  scene.push_back( scene_keypoints [ good_matches[i].trainIdx ].pt );
			}//: for

			// Find homography between corresponding points.
			Mat H = findHomography( obj, scene, CV_RANSAC );

			// Get the corners from the detected "object hypothesis".
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cv::Point2f(0,0);
			obj_corners[1] = cv::Point2f( models_imgs[m].cols, 0 );
			obj_corners[2] = cv::Point2f( models_imgs[m].cols, models_imgs[m].rows );
			obj_corners[3] = cv::Point2f( 0, models_imgs[m].rows );
			std::vector<Point2f> hypobj_corners(4);

			// Transform corners with found homography.
			perspectiveTransform( obj_corners, hypobj_corners, H);
			
			// Verification: check resulting shape of object hypothesis.
			// Compute "center of mass".
			cv::Point2f center = (hypobj_corners[0] + hypobj_corners[1] + hypobj_corners[2] + hypobj_corners[3])*.25;
			std::vector<double> angles(4);
			cv::Point2f tmp ;
			// Compute angles.
			for (int i=0; i<4; i++) {
				tmp = (hypobj_corners[i] - center);
				angles[i] = atan2(tmp.y,tmp.x);
				CLOG(LDEBUG)<< tmp << " angle["<<i<<"] = "<< angles[i];
			}//: if


			// Find smallest element.
			int imin = -1;
			double amin = 1000;
			for (int i=0; i<4; i++)
				if (amin > angles[i]) {
					amin = angles[i];
					imin = i;
				}//: if

			// Reorder table.
			for (int i=0; i<imin; i++) {
				angles.push_back (angles[0]);
				angles.erase(angles.begin());
			}//: for

			for (int i=0; i<4; i++) {
				CLOG(LDEBUG)<< "reordered angle["<<i<<"] = "<< angles[i];
			}//: if

			cv::Scalar colour;
			double score = (double)good_matches.size()/models_keypoints [m].size();
			// Check dependency between corners.
			if ((angles[0] < angles[1]) && (angles[1] < angles[2]) && (angles[2] < angles[3])) {
				// Order is ok.
				colour = Scalar(0, 255, 0);
				CLOG(LINFO)<< "Model ("<<m<<"): keypoints "<< models_keypoints [m].size()<<" corrs = "<< good_matches.size() <<" score "<< score << " VALID";
				// Store the model in a list in proper order.
				storeObjectHypothesis(models_names[m], center, hypobj_corners, score);

			} else {
				// Hypothesis not valid.
				colour = Scalar(0, 0, 255);
				CLOG(LINFO)<< "Model ("<<m<<"): keypoints "<< models_keypoints [m].size()<<" corrs = "<< good_matches.size() <<" score "<< score << " REJECTED";
			}//: else
				

			if (m == prop_returned_model_number) {
				Mat img_matches2;
				// Draw good matches.
				drawMatches( models_imgs[m], models_keypoints[m], scene_img, scene_keypoints,
					     good_matches, img_matches2, Scalar::all(-1), Scalar::all(-1),
					     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
				// Draw the object as lines, with center and top left corner indicated.
				line( img_matches2, hypobj_corners[0] + Point2f( models_imgs[m].cols, 0), hypobj_corners[1] + Point2f( models_imgs[m].cols, 0), colour, 4 );
				line( img_matches2, hypobj_corners[1] + Point2f( models_imgs[m].cols, 0), hypobj_corners[2] + Point2f( models_imgs[m].cols, 0), colour, 4 );
				line( img_matches2, hypobj_corners[2] + Point2f( models_imgs[m].cols, 0), hypobj_corners[3] + Point2f( models_imgs[m].cols, 0), colour, 4 );
				line( img_matches2, hypobj_corners[3] + Point2f( models_imgs[m].cols, 0), hypobj_corners[0] + Point2f( models_imgs[m].cols, 0), colour, 4 );
				circle( img_matches2, center + Point2f( models_imgs[m].cols, 0), 2, colour, 4);
				circle( img_matches2, hypobj_corners[0] + Point2f( models_imgs[m].cols, 0), 2, Scalar(255, 0, 0), 4);
				out_img_good_correspondences.write(img_matches2);

			}//: if
		}//: for
		
		Mat img_object = scene_img.clone();
		if (recognized_names.size() == 0) {
			CLOG(LWARNING)<< "None of the models was not properly recognized in the image";
		} else {
			
			for (int h=0; h<recognized_names.size(); h++) {
				// Draw the final object - as lines, with center and top left corner indicated.
				line( img_object, recognized_corners[h][0], recognized_corners[h][1], Scalar(0, 255, 0), 4 );
				line( img_object, recognized_corners[h][1], recognized_corners[h][2], Scalar(0, 255, 0), 4 );
				line( img_object, recognized_corners[h][2], recognized_corners[h][3], Scalar(0, 255, 0), 4 );
				line( img_object, recognized_corners[h][3], recognized_corners[h][0], Scalar(0, 255, 0), 4 );
				circle( img_object, recognized_centers[h], 2, Scalar(0, 255, 0), 4);
				circle( img_object, recognized_corners[h][0], 2, Scalar(255, 0, 0), 4);
				CLOG(LNOTICE)<< "Hypothesis (): model: "<< recognized_names[h]<< " score: "<< recognized_scores[h];
			}//: for
		}//: else
		// Write image to port.
		out_img_object.write(img_object);
	} catch (...) {
		CLOG(LERROR) << "onNewImage failed";
	}//: catch
}


} //: namespace TORecognize
} //: namespace Processors
