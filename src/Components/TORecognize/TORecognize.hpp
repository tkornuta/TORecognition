/*!
 * \file
 * \brief 
 * \author Tomek Kornuta,,,
 */

#ifndef TORECOGNIZE_HPP_
#define TORECOGNIZE_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "EventHandler2.hpp"

#include "Types/KeyPoints.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>


namespace Processors {
namespace TORecognize {

using namespace cv;

/*!
 * \class TORecognize
 * \brief TORecognize processor class.
 *
 * TORecognize processor.
 */
class TORecognize: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	TORecognize(const std::string & name = "TORecognize");

	/*!
	 * Destructor
	 */
	virtual ~TORecognize();

	/*!
	 * Prepare components interface (register streams and handlers).
	 * At this point, all properties are already initialized and loaded to 
	 * values set in config file.
	 */
	void prepareInterface();

protected:

	/*!
	 * Connects source to given device.
	 */
	bool onInit();

	/*!
	 * Disconnect source from device, closes streams, etc.
	 */
	bool onFinish();

	/*!
	 * Start component
	 */
	bool onStart();

	/*!
	 * Stop component
	 */
	bool onStop();

	/*!
	 * Event handler function.
	 */
	void onNewImage();

	/// Input data stream
	Base::DataStreamIn <cv::Mat> in_img;

	/// Output data stream - image.
	Base::DataStreamOut <cv::Mat> out_img;

private:
	// Keypoint detector - FAST.
        cv::FastFeatureDetector detector;
	// Feature descriptor - FREAK.
        cv::FREAK extractor;
	// Matcher.
	cv::BFMatcher matcher;
	// "Database" of objects.
        cv::Mat gray_object;

        std::vector<KeyPoint> keypoints_object;
        cv::Mat descriptors_object;
};

} //: namespace TORecognize
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("TORecognize", Processors::TORecognize::TORecognize)

#endif /* TORECOGNIZE_HPP_ */
