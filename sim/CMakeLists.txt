cmake_minimum_required(VERSION 2.8.6)
project(sim)



SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#	FORCE
	)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${DART_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB srcs "*.h" "*.cpp")
list(FILTER srcs EXCLUDE REGEX "BVHToXML.cpp")
list(FILTER srcs EXCLUDE REGEX "swing_cleanup.cpp")

add_library(sim ${srcs} )
target_link_libraries(sim ${DART_LIBRARIES} ${Boost_LIBRARIES} ${TinyXML_LIBRARIES} ${PYTHON_LIBRARIES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#add_executable(xmlGen BVHToXML.cpp)
#target_link_libraries(xmlGen ${TinyXML_LIBRARIES})

add_executable(cleanup swing_cleanup.cpp ${srcs})
target_link_libraries(cleanup ${DART_LIBRARIES} ${Boost_LIBRARIES} ${TinyXML_LIBRARIES} ${PYTHON_LIBRARIES})
