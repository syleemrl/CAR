#ifndef __META_CONTROLLER_H
#define __META_CONTROLLER_H
#include "SubController.h"
#include "EnemyKinController.h"

namespace DPhy
{

class MetaController
{
public:
	MetaController();
	
	Eigen::VectorXd GetCurrentRefPositions() {return mTargetPositions;}
	Eigen::VectorXd GetCurrentSimPositions() {return mCharacter->GetSkeleton()->getPositions();}
	double GetCurrentPhase();
	Eigen::Vector3d GetCOM() {return mCharacter->GetSkeleton()->getCOM(); }

	void SwitchController(std::string type, Eigen::VectorXd target, int frame=0, bool isEnemy=false);
	void SetAction();
	void LoadControllers();
	void AddSubController(SubController* new_sc){mSubControllers[new_sc->mType]= new_sc;}
	void Reset();
	void Step();
	std::string GetNextAction();
	void SaveAsBVH(std::string filename, std::vector<Eigen::VectorXd> record);

	int AddNewEnemy(Eigen::VectorXd d);
	std::vector<int> GetCurrentEnemyIdxs() {return curEnemyList;}
	Eigen::VectorXd GetEnemyPositions(int i);
	void ToggleTargetPhysicsMode();
	void IsActive();
	void SwitchMainTarget();
	bool IsEnemyPhysics();
	void ClearFallenEnemy();
	void SaveAll(std::string filename);
	SubController* mPrevController;	
	SubController* mCurrentController=nullptr;
	SubController* mCurrentEnemyController=nullptr;

	std::pair<std::string, double> mWaiting;	

	std::map<std::string, SubController*> mSubControllers;
	std::map<std::string, SubController*> mSubControllersEnemy;

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
	std::vector<std::vector<Eigen::VectorXd>> mRecordEnemyPosition;
	std::vector<double> mRecordEnemyTiming;
	std::vector<Eigen::Vector3d> mHitPoints;
	bool mIsWaiting=false;
	bool mActionSelected;
	Eigen::VectorXd mNextTarget;

	std::random_device mRD;
	std::mt19937 mMT;
	std::uniform_real_distribution<double> mUniform;

	std::string mPrevAction="";
	std::vector<int> curEnemyList;
	std::vector<EnemyKinController*> mEnemyController;

	int mTargetEnemyIdx=0;
};
} 

#endif