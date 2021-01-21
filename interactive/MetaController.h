#ifndef __META_CONTROLLER_H
#define __META_CONTROLLER_H
#include "SubController.h"

namespace DPhy
{

class MetaController
{
public:
	MetaController();
	
	Eigen::VectorXd GetCurrentRefPositions() {return mTargetPositions;}
	Eigen::VectorXd GetCurrentSimPositions() {return mCharacter->GetSkeleton()->getPositions();}

	void SwitchController(std::string type, int frame=0);
	void LoadControllers();
	void AddSubController(SubController* new_sc){mSubControllers[new_sc->mType]= new_sc;}
	void Reset();
	void Step();

	SubController* mPrevController;	
	SubController* mCurrentController;
	std::pair<std::string, double> mWaiting;	

	std::map<std::string, SubController*> mSubControllers;

	//World-related stuff
	dart::simulation::WorldPtr mWorld;

	int mControlHz;
	int mSimulationHz;
	int mSimPerCon;

	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;

	double mTotalSteps;
	
	Eigen::VectorXd mTargetPositions;
	std::vector<Eigen::VectorXd> mRecordPosition;
	std::vector<Eigen::Vector3d> mHitPoints;
	bool mIsWaiting=false;
};
} 

#endif