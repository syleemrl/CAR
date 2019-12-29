#ifndef __DEEP_PHYSICS_CONTROLLER_H__
#define __DEEP_PHYSICS_CONTROLLER_H__
#include "dart/dart.hpp"
#include "BVH.h"
#include "CharacterConfigurations.h"
#include "SkeletonBuilder.h"
#include "Functions.h"
#include "ReferenceManager.h"
namespace DPhy
{
class Character;

/**
*
* @brief World class expresses individual virtual world which contains character and ground.
* @details Character and ground are agent and ground information respectively. Each world contains both of them and also able to interactive environment status with super level.
* 
*/
class Controller
{
public:
Controller(std::string motion, bool record=false, bool use_bvh=true);

	void Step();
	void UpdateReward();
	void UpdateAdaptiveReward();
	void UpdateTerminalInfo();
	void Reset(bool RSI=true);
	void SetReference(std::string motion);
	bool FollowBvh();
	bool IsTerminalState() {return this->mIsTerminal; }
	bool IsNanAtTerminal() {return this->mIsNanAtTerminal;}
	bool IsTimeEnd(){
		if(this->terminationReason == 8)
			return true;
		else
			return false;
	}
	int GetNumState();
	int GetNumAction();
	Eigen::VectorXd GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel);

	bool CheckCollisionWithGround(std::string bodyName);
	Eigen::VectorXd GetState();
	void SetAction(const Eigen::VectorXd& action);
	double GetReward() {return mRewardParts[0]; }
	std::vector<double> GetRewardByParts() {return mRewardParts; }
	const dart::simulation::WorldPtr& GetWorld() {return mWorld;}

	double GetTimeElapsed(){return this->mTimeElapsed;}
	double GetCurrentFrame(){return this->mCurrentFrame;}
	double GetCurrentLength() {return this->mCurrentFrame - this->mStartFrame; }
	double GetStartFrame(){ return this->mStartFrame; }

	const dart::dynamics::SkeletonPtr& GetSkeleton();
	const dart::dynamics::SkeletonPtr& GetRefSkeleton();

	void RescaleCharacter(double w0, double w1);
	void SaveTrainedData(std::string directory);

	void UpdateGRF(std::vector<std::string> joints);
	std::vector<Eigen::VectorXd> GetGRF();
	void SaveStepInfo();
	Eigen::VectorXd GetPositions(int idx) { return this->mRecordPosition[idx]; }
	Eigen::Vector3d GetCOM(int idx) { return this->mRecordCOM[idx]; }
	Eigen::VectorXd GetVelocities(int idx) { return this->mRecordVelocity[idx]; }
	double GetTime(int idx) { return this->mRecordTime[idx]; }

	int GetRecordSize() { return this->mRecordPosition.size(); }
	std::pair<bool, bool> GetFootContact(int idx) { return this->mRecordFootContact[idx]; }
	std::tuple<double, double, double> GetRescaleParameter() { return mRescaleParameter; }
	
	void computeEnergyConservation();

protected:
	dart::simulation::WorldPtr mWorld;
	double w_p,w_v,w_com,w_ee,w_srl;
	int mTimeElapsed;
	double mStartFrame;
	double mCurrentFrame; // for discrete ref motion

	int mControlHz;
	int mSimulationHz;
	int mSimPerCon;
	double mStep;
	bool mUseBVH;

	Character* mCharacter;
	Character* mRefCharacter;
	ReferenceManager* mReferenceManager;
	dart::dynamics::SkeletonPtr mGround;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mTargetContacts;

	Eigen::VectorXd mPDTargetPositions;
	Eigen::VectorXd mPDTargetVelocities;

	Eigen::VectorXd mActions;

	Eigen::Vector3d mTargetCOMvelocity;
	double mAdaptiveCOM;
	int mAdaptiveStep;

	std::vector<std::string> mInterestedBodies;
	std::vector<std::string> mRewardBodies;
	std::vector<std::string> mEndEffectors;

	std::vector<double> mRewardParts;
	// for foot collision, left, right foot, ground
	std::unique_ptr<dart::collision::CollisionGroup> mCGEL, mCGER, mCGL, mCGR, mCGG, mCGHR, mCGHL; 

	std::vector<Eigen::VectorXd> mRecordPosition;
	std::vector<Eigen::VectorXd> mRecordVelocity;
	std::vector<Eigen::Vector3d> mRecordCOM;
	std::vector<double> mRecordEnergy;
	std::vector<double> mRecordWork;
	std::vector<double> mRecordDCOM;
	std::vector<Eigen::VectorXd> mRecordTorque;

	std::vector<std::pair<bool, bool>> mRecordFootContact;
	bool mIsTerminal;
	bool mIsNanAtTerminal;
	bool mRecord;
	std::tuple<bool, double, double> mDoubleStanceInfo;
	std::tuple<double, double, double> mRescaleParameter;
	std::vector<std::string> mGRFJoints;
	std::vector<double> mRecordTime;
	std::vector<double> mRecordDTime;

	std::vector<Eigen::Vector6d> mRecordCOMVelocity;
	std::vector<Eigen::Vector3d> mRecordCOMPositionRef;
	std::pair<double, int> mInputVelocity;
	int mNumState, mNumAction;

	int terminationReason;

	std::vector<std::vector<Eigen::VectorXd>> GRFs;
	std::shared_ptr<dart::collision::DARTCollisionDetector> mGroundCollisionChecker;	

	std::vector<Eigen::VectorXd> mTargetTorques;
	Eigen::VectorXd mTorqueMean;
	Eigen::VectorXd mTorqueMin;
	Eigen::VectorXd mTorqueMax;

	Eigen::VectorXd mTorqueSig;
};
}
#endif
