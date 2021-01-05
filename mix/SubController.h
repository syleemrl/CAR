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

class MetaController;
class SubController
{
public:
	SubController(){}
	SubController(std::string type, MetaController* mc, std::string motion, std::string ppo, std::string reg);

	std::string mType;

	p::object 						mPPO;
	DPhy::ReferenceManager*			mReferenceManager;
	p::object 						mRegression;
	DPhy::RegressionMemory* 		mRegressionMemory;

	Eigen::VectorXd mParamGoal;

	bool virtual IsTerminalState()=0;
	bool virtual Step()=0;
	void virtual reset(double frame=0, double frameOnPhase=0)=0;
	void virtual setCurObject(dart::dynamics::SkeletonPtr object){mCurObject = object;}

	Eigen::VectorXd GetParamGoal(){return mParamGoal;}

	std::string mMotion;
	MetaController* mMC;
	bool mUseReg= false;

	dart::dynamics::SkeletonPtr mCurObject;

	double mCurrentFrame;
	double mCurrentFrameOnPhase;
};


////// REALIZATION (CHILD SUBCONTROLLER) //////

class FW_JUMP_Controller : public SubController
{
public:
	FW_JUMP_Controller(){}
	FW_JUMP_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg);

	bool virtual IsTerminalState();
	bool virtual Step();
	void virtual reset(double frame=0,double frameOnPhase=0);
};


class WALL_JUMP_Controller : public SubController
{
public:
	WALL_JUMP_Controller(){}
	WALL_JUMP_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg);

	bool virtual IsTerminalState();
	bool virtual Step();
	void virtual reset(double frame=0,double frameOnPhase=0);


	dart::constraint::BallJointConstraintPtr leftHandConstraint= nullptr;
	dart::constraint::BallJointConstraintPtr rightHandConstraint= nullptr;

	bool left_detached = false;
	bool right_detached = false;

	void attachHandToBar(bool left, Eigen::Vector3d offset);
	void removeHandFromBar(bool left);

};

}// end namespace DPhy

#endif