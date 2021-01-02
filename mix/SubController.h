#ifndef __SUB_CONTROLLER_H
#define __SUB_CONTROLLER_H


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
#pragma pop_macro("slots")

namespace p = boost::python;
namespace np = boost::python::numpy;


namespace DPhy
{
enum CTR_TYPE{
	FW_JUMP,
	WALL_JUMP,
	RUN_SWING,
	BOX_CLIMB
};

class MetaController;
class SubController
{
public:
	SubController(){}
	SubController(CTR_TYPE type, MetaController* mc, std::string motion, std::string ppo, std::string reg);

	CTR_TYPE mType;

	p::object 						mPPO;
	DPhy::ReferenceManager*			mReferenceManager;
	p::object 						mRegression;
	DPhy::RegressionMemory* 		mRegressionMemory;

	Eigen::VectorXd mParamGoal;

	bool virtual IsTerminalState()=0;
	bool virtual Step()=0;

	Eigen::VectorXd GetParamGoal(){return mParamGoal;}

	std::string mMotion;
	MetaController* mMC;
	bool mUseReg= false;
};

}// end namespace DPhy

//void Controller::loadScene(){

// 	std::string scene_path = std::string(CAR_DIR)+std::string("/scene/") + std::string(SCENE) + std::string(".xml");
// 	mSceneObjects = std::vector<dart::dynamics::SkeletonPtr>();
// 	SkeletonBuilder::loadScene(scene_path, mSceneObjects);
// 	for(auto obj: mSceneObjects) this->mWorld->addSkeleton(obj);

// 	this->mloadScene = true;
// }

#endif