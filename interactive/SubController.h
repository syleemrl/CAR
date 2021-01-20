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
	SubController(std::string type, std::string motion, std::string ppo);

	std::string mType;

	p::object 						mPPO;
	DPhy::ReferenceManager*			mReferenceManager;
	DPhy::RegressionMemory* 		mRegressionMemory;

	bool mIsParametric;
	Eigen::VectorXd mParamGoal;

	bool Step(dart::simulation::WorldPtr world, Character* character);
	void Synchronize(Character* character, double frame=0);
	bool virtual Synchronizable(std::string)=0;

	Eigen::VectorXd GetState(Character* character);
	Eigen::VectorXd GetEndEffectorStatePosAndVel(Character* character, Eigen::VectorXd pos, Eigen::VectorXd vel);
	Eigen::VectorXd GetCurrentRefPositions() { return mTargetPositions; }
	double mCurrentFrame;
	double mPrevFrame;

	double mCurrentFrameOnPhase;

	int mSimPerCon;
	
	int numStates;
	int numActions;
	int mInterestedDof;
	double mAdaptiveStep;

	Eigen::VectorXd mPrevTargetPositions;
	Eigen::VectorXd mTargetPositions;
	std::vector<std::string> mEndEffectors;

};


////// REALIZATION (CHILD SUBCONTROLLER) //////

class PUNCH_Controller : public SubController
{
public:
	PUNCH_Controller(){}
	PUNCH_Controller(std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string);
};


class IDLE_Controller : public SubController
{
public:
	IDLE_Controller(){}
	IDLE_Controller(std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string);

};


class BLOCK_Controller : public SubController
{
public:
	BLOCK_Controller(){}
	BLOCK_Controller( std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string);

};
class PIVOT_Controller : public SubController
{
public:
	PIVOT_Controller(){}
	PIVOT_Controller( std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string);

};

class KICK_Controller : public SubController
{
public:
	KICK_Controller(){}
	KICK_Controller(std::string motion, std::string ppo);
	
	bool virtual Synchronizable(std::string);
};


}// end namespace DPhy

#endif