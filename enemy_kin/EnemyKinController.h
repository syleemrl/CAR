#ifndef __ENMY_KIN_CONTROLLER_H
#define __ENMY_KIN_CONTROLLER_H



#include "Character.h"
#include "MultilevelSpline.h"
#include <boost/filesystem.hpp>
#include <Eigen/QR>
#include <fstream>
#include <numeric>
#include <algorithm>

#pragma push_macro("slots")
#undef slots
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "Camera.h"
#include "ReferenceManager.h"
#include "RegressionMemory.h"
#include "GLfunctions.h"
#include "DART_interface.h"
#include "Controller.h"
#include "Functions.h"
#pragma pop_macro("slots")

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace DPhy
{

class EnemyKinController
{
public:
	EnemyKinController();

	void Step();
	void Reset();
	Character* mCharacter;	
	std::vector<Eigen::VectorXd> mRecordPosition;

	Eigen::Vector3d target;
	double distance;

	Character* mCharacter_main_tmp;	

	std::string mCurrentMotion;
	std::string mNextMotion;
	int mCurrentFrameOnPhase;

	std::map<std::string, int> mMotionFrames;
	DPhy::ReferenceManager* mReferenceManager;

	Eigen::VectorXd GetPosition();
	void Step(Eigen::VectorXd main_p);

	Eigen::Isometry3d mAlign;
	void calculateAlign();

	int mTotalFrame;
};
} 

#endif