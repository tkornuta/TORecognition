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
	// Register data streams, events and event handlers HERE!

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



} //: namespace TORecognize
} //: namespace Processors
