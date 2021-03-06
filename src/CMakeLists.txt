SET(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(.)

# ##############################################################################
# Find required libraries
# ##############################################################################

# Find Boost, at least ver. 1.41
FIND_PACKAGE(Boost 1.41.0 REQUIRED COMPONENTS system thread filesystem date_time)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# Find another necessary libraries
# OpenCV library
FIND_PACKAGE( OpenCV REQUIRED )

# CvCoreTypes DCL
DISCODE_FIND_DCL(CvCoreTypes)

# Set variable with list of all libraries common for this DCL
#SET(DisCODe_LIBRARIES ${DisCODe_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
SET(DisCODe_LIBRARIES ${DisCODe_LIBRARIES} ${Boost_LIBRARIES})


# Check, whether all necessary libraries are linked
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs") 

# ##############################################################################
# Add source directories
# ##############################################################################

# Components
ADD_SUBDIRECTORY(Components)

# Prepare config file to use from another DCLs
CONFIGURE_FILE(TORecognitionConfig.cmake.in ${CMAKE_INSTALL_PREFIX}/TORecognitionConfig.cmake @ONLY)
