cmake_minimum_required(VERSION 2.8.6)
project(render_qt)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#	FORCE
	)
set(CMAKE_AUTOMOC ON)

link_directories(../sim/)
include_directories(../sim/)
link_directories(../network/)
include_directories(../network/)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5OpenGL CONFIG REQUIRED)
link_libraries(stdc++fs)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${DART_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${TinyXML_INCLUDE_DIRS})

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB srcs "*.h" "*.cpp")

add_executable(render_qt ${srcs})
target_link_libraries(render_qt Qt5::Widgets Qt5::OpenGL ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut GLEW sim ${TinyXML_LIBRARIES})

file (GLOB lib_srcs
	Camera.cpp
	DART_interface.cpp
	GLfunctions.cpp
	)
add_library(render_lib ${lib_srcs} )
target_link_libraries(render_lib ${DART_LIBRARIES} ${Boost_LIBRARIES} ${TinyXML_LIBRARIES} ${PYTHON_LIBRARIES})