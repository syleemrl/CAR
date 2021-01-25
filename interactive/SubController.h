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
#include "RegressionMemory.h"
#include "ReferenceManager.h"
#include "Controller.h"
#include "Functions.h"
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

	bool Step();
	std::pair<Eigen::VectorXd, Eigen::VectorXd> GetPDTarget(Character* character);
	void Synchronize(Character* character, Eigen::VectorXd endPosition, double frame, int debug=0);
	bool virtual Synchronizable(std::string next)=0;
	void virtual SetAction(Eigen::VectorXd tp)=0;

	Eigen::VectorXd GetState(Character* character, int debug=0);
	Eigen::VectorXd GetEndEffectorStatePosAndVel(Character* character, Eigen::VectorXd pos, Eigen::VectorXd vel);
	Eigen::VectorXd GetCurrentRefPositions() { return mTargetPositions; }
	bool IsEnd() { return mEndofMotion; }
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

	bool mEndofMotion;
	bool mActionSelected;
	Eigen::VectorXd mPrevEndPos;
	int mBlendInterval;
};


////// REALIZATION (CHILD SUBCONTROLLER) //////

class PUNCH_Controller : public SubController
{
public:
	PUNCH_Controller(){}
	PUNCH_Controller(std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string next);
	void virtual SetAction(Eigen::VectorXd tp);
};

class PUNCH_ENEMY_Controller : public SubController
{
public:
	PUNCH_ENEMY_Controller(){}
	PUNCH_ENEMY_Controller(std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string next);
	void virtual SetAction(Eigen::VectorXd tp);
};

class IDLE_Controller : public SubController
{
public:
	IDLE_Controller(){}
	IDLE_Controller(std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string next);
	void virtual SetAction(Eigen::VectorXd tp) {}
};


class DODGE_Controller : public SubController
{
public:
	DODGE_Controller(){}
	DODGE_Controller( std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string next);
	void virtual SetAction(Eigen::VectorXd tp);
};
class PIVOT_Controller : public SubController
{
public:
	PIVOT_Controller(){}
	PIVOT_Controller( std::string motion, std::string ppo);

	bool virtual Synchronizable(std::string next);
	void virtual SetAction(Eigen::VectorXd tp);
};

class KICK_Controller : public SubController
{
public:
	KICK_Controller(){}
	KICK_Controller(std::string motion, std::string ppo);
	
	bool virtual Synchronizable(std::string next);
	void virtual SetAction(Eigen::VectorXd tp);
};


}// end namespace DPhy

#endif