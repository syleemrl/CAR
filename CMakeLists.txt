cmake_minimum_required(VERSION 2.8.6)
project(CAR)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++17)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-lm)

add_definitions(-DCAR_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake_module")
find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt collision-bullet collision-ode CONFIG)	
find_package(Boost 1.67.0 COMPONENTS filesystem python36 numpy36 program_options)
IF(NOT Boost_FOUND)
	find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3 program_options)
ENDIF()
find_package(PythonLibs 3.6 REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(TinyXML REQUIRED)

add_subdirectory( sim )
add_subdirectory( network )
#add_subdirectory( render )
add_subdirectory( render_qt )
