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

	/// Output data stream - image containing all correspondences.
	Base::DataStreamOut <cv::Mat> out_img_all_correspondences;

	/// Output data stream - image containing only selected (good) correspondences.
	Base::DataStreamOut <cv::Mat> out_img_good_correspondences;

	/// Output data stream - image with object.
	Base::DataStreamOut <cv::Mat> out_img_object;

	/// Property - number of the model that will be returned on output image (along with features and correspondences).
	Base::Property<int> prop_returned_model_number;

	/// Property - filename (including directory).
	Base::Property<std::string> prop_filename;

	///  Propery - if set, reads model image at start.
	Base::Property<bool> prop_read_on_init;

	/// Property - limit of returned/displayed recognized objects.
	Base::Property<int> prop_recognized_object_limit;


private:

	// Vector of images constituting the consecutive models.
        std::vector<cv::Mat> models_imgs;

	/// Vector of keypoints of consecutive models.
        std::vector<std::vector<cv::KeyPoint> > models_keypoints;

	/// Vector of descriptors of consecutive models.
        std::vector<cv::Mat> models_descriptors;

	/// Vector of names of consecutive models.
        std::vector<std::string> models_names;



	/// Vector containing names of recognized objects.
	std::vector<std::string> recognized_names;

	/// Vector containing centers of recognized objects (image coordinates).
	std::vector<cv::Point2f> recognized_centers;

	/// Vector containing quadruples of corners of recognized objects (image coordinates).
	std::vector<std::vector<cv::Point2f> > recognized_corners;

	/// Vector containint scores of recognized objects.
	std::vector<double> recognized_scores;

	/// Stores recognized hypothesis in proper order - from the one with the highest score to the one with lowest.
	void storeObjectHypothesis(std::string name_, cv::Point2f center_, std::vector<cv::Point2f> corners_, double score_);


	/// Sets load_model_flag when the used presses button.
	void onLoadModelButtonPressed();
	
	/// Flag used for loading models.
	bool load_model_flag;

	/// Re-load the models from files, detect and extract their features.
	void loadModels();

	/// Load a single model from file indicated by function parameter.
	void loadSingleModel(std:: string filename_, std::string name_);


	/// Loads image from file.
	bool loadImage(const std::string filename_, cv::Mat & image_);

	/// Returns keypoint with descriptors extracted from image.
	bool extractFeatures(const cv::Mat image_, std::vector<KeyPoint> & keypoints_, cv::Mat & descriptors_);



	/// Keypoint detector.
	Ptr<FeatureDetector> detector;

	/// Sets the keypoint detector according to the current selection (see: prop_detector_type).
	void setKeypointDetector();
	
	///  Propery - type of keypoint detector: 0 - FAST (default), 1 - STAR , 2 - SIFT , 3 - SURF , 4 - ORB , 5 - BRISK , 6 - MSER , 7 - GFTT , 8 - GFTT, 9 - Dense, 10 - SimpleBlob
	Base::Property<int> prop_detector_type;

	/// Variable denoting current detector type - used for dynamic switching between detectors.
	int current_detector_type;



	/// Feature descriptor.
        cv::Ptr<cv::DescriptorExtractor> extractor;

	/// Sets the extreactor according to the current selection (see: prop_extractor_type).
	void setDescriptorExtractor();
	
	///  Propery - type of feature descriptor: 0 - SIFT (default), 1 - SURF, 2 - BRIEF, 3 - BRISK, 4 - ORB, 5 - FREAK
	Base::Property<int> prop_extractor_type;

	/// Variable denoting current extractor type - used for dynamic switching between extractors.
	int current_extractor_type;



	// Matcher.
	DescriptorMatcher* matcher;
	
	/// Sets the matcher according to the current selection (see: prop_matcher_type).
	void setDescriptorMatcher();
	
	///  Propery - type of descriptor matcher: 0 - BF with L2 (default), 1 - BF with L2 and crosscheck, 2 - BF with Hamming, 3 - BF with Hamming and crosscheck, 4 - FLANN with L2, 5 - FLANN with LSH
	Base::Property<int> prop_matcher_type;

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
