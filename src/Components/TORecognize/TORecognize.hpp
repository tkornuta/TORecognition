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



namespace Processors {
namespace TORecognize {

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


	

};

} //: namespace TORecognize
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("TORecognize", Processors::TORecognize::TORecognize)

#endif /* TORECOGNIZE_HPP_ */
