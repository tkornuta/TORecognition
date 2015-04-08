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
		Base::Component(name)  {

}

TORecognize::~TORecognize() {
}

void TORecognize::prepareInterface() {
	// Register handlers with their dependencies.
    registerHandler("onNewImage", boost::bind(&TORecognize::onNewImage, this));
    addDependency("onNewImage", &in_img);

	// Input and output data streams.
    registerStream("in_img", &in_img);
    registerStream("out_img", &out_img);
	//registerStream("out_features", &out_features);
}

bool TORecognize::onInit() {
	// Load the "database" of objects.
        Mat img_object = imread("/home/tkornuta/discode_ecovi/DCL/ecovi/data/tea_covers/lipton_green_tea_citrus.jpg");

	// Transform to grayscale.
        cvtColor(img_object, gray_object, COLOR_BGR2GRAY);

        //-- Step 1: Detect the keypoints of the object.
        detector.detect( gray_object, keypoints_object );

        //-- Step 2: Calculate descriptors (feature vectors) of the object.
        extractor.compute( gray_object, keypoints_object, descriptors_object );

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

void TORecognize::onNewImage()
{
	CLOG(LTRACE) << "TORecognize::onNewImage\n";
	try {
	// Input: a grayscale image.
        cv::Mat img_scene = in_img.read();
        cv::Mat gray_scene;
        cvtColor(img_scene, gray_scene, COLOR_BGR2GRAY);

//        out_img.write(gray_scene);


//        if( !gray_object.data || !gray_scene.data )
//        { std::cout<< " --(!) Error reading images " << std::endl; return ; }

        //-- Step 1: Detect the keypoints of the scene.
        std::vector<KeyPoint> keypoints_scene;
        detector.detect( gray_scene, keypoints_scene );

        //-- Step 2: Calculate descriptors (feature vectors) of the scene.
        Mat descriptors_scene;
        extractor.compute( gray_scene, keypoints_scene, descriptors_scene );


        //-- Step 3: Matching descriptor vectors using BF matcher with L2 metric.
//        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptors_object, descriptors_scene, matches );

	// Draw all matches.
        Mat img_matches1;
        drawMatches( gray_object, keypoints_object, gray_scene, keypoints_scene,
                     matches, img_matches1, Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        out_img.write(img_matches1);


/*        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_object.rows; i++ )
        { double dist = matches[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
        }

        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );

        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector< DMatch > good_matches;

        for( int i = 0; i < descriptors_object.rows; i++ )
        { if( matches[i].distance < 3*min_dist )
           { good_matches.push_back( matches[i]); }
        }

        Mat img_matches2;
        drawMatches( gray_object, keypoints_object, gray_scene, keypoints_scene,
                     good_matches, img_matches2, Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        out_img.write(img_matches2);*/

/*
        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;

        for( int i = 0; i < good_matches.size(); i++ )
        {
          //-- Get the keypoints from the good matches
          obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
          scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }

        Mat H = findHomography( obj, scene, CV_RANSAC );

        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
        obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
        std::vector<Point2f> scene_corners(4);

        perspectiveTransform( obj_corners, scene_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f( gray_object.cols, 0), scene_corners[1] + Point2f( gray_object.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f( gray_object.cols, 0), scene_corners[2] + Point2f( gray_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f( gray_object.cols, 0), scene_corners[3] + Point2f( gray_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f( gray_object.cols, 0), scene_corners[0] + Point2f( gray_object.cols, 0), Scalar( 0, 255, 0), 4 );

        //-- Show detected matches
        //imshow( "Good Matches & Object detection", img_matches );
        out_img.write(img_matches);*/

	} catch (...) {
		CLOG(LERROR) << "TORecognize::onNewImage failed\n";
	}
}


} //: namespace TORecognize
} //: namespace Processors
