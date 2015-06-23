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

	/// Property - filename (including directory).
	Base::Property<std::string> prop_filename;

	///  Propery - if set, reads model image at start.
	Base::Property<bool> prop_read_on_init;

	///  Propery - type of descriptor matcher: 0 - BF with L2 (default), 1 - BF with Hamming, 2 - FLANN with L2, 3 - FLANN with LSH
	Base::Property<int> prop_matcher_type;

private:
	// Keypoint detector - FAST.
        cv::FastFeatureDetector detector;

	// Feature descriptor - FREAK.
        cv::FREAK extractor;

	// Matcher.
	DescriptorMatcher* matcher;
	
	// "Object model" - colour image.
        cv::Mat model_img;

	/// Vector of model keypoints.
        std::vector<KeyPoint> model_keypoints;

	/// Vector of model descriptors.
        cv::Mat model_descriptors;

	/// Loads image from file.
	bool loadImage(const std::string filename_, cv::Mat & image_);

	/// Returns keypoint with descriptors extracted from image.
	bool extractFeatures(const cv::Mat image_, std::vector<KeyPoint> & keypoints_, cv::Mat & descriptors_);

	/// Re-load the model when the used presses button.
	void onLoadModelButtonPressed();

	/// Sets the matcher according to the current selection (see: prop_matcher_type).
	void setMatcher();
	
	/// Variable denoting current matcher type - used for dynamic switching between matchers.
	int current_matcher_type;

};

} //: namespace TORecognize
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("TORecognize", Processors::TORecognize::TORecognize)

#endif /* TORECOGNIZE_HPP_ */
