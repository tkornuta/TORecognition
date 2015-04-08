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
	//registerStream("out_features", &out_features);
}

bool TORecognize::onInit() {

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
        cv::Mat img = in_img.read();
        cv::Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        //-- Step 1: Detect the keypoints using FAST Detector.
        std::vector<KeyPoint> keypoints;
        //cv::FAST(gray,keypoints,10);
        cv::FastFeatureDetector detector(10);
        detector.detect( gray, keypoints );



		//-- Step 2: Calculate descriptors (feature vectors) using Freak descriptor.
        cv::FREAK extractor;
        cv::Mat descriptors;
        extractor.compute( gray, keypoints, descriptors);

		// Write features to the output.
	    Types::KeyPoints kpts(keypoints);
		//out_features.write(features);

		// Write descriptors to the output.
		//out_descriptors.write(descriptors);
	} catch (...) {
		LOG(LERROR) << "CvFreak::onNewImage failed\n";
	}
}


} //: namespace TORecognize
} //: namespace Processors
