#ifndef __DEEP_PHYSICS_CONTROLLER_H__
#define __DEEP_PHYSICS_CONTROLLER_H__
#include "dart/dart.hpp"
#include "BVH.h"
#include "CharacterConfigurations.h"
#include "SkeletonBuilder.h"
#include "Functions.h"
#include "ReferenceManager.h"
#include <tuple>

namespace DPhy
{
/**
*
* @brief World class expresses individual virtual world which contains character and ground.
* @details Character and ground are agent and ground information respectively. Each world contains both of them and also able to interactive environment status with super level.
* 
*/
class Controller
{
public:
Controller(ReferenceManager* ref, bool adaptive=true, bool record=false, int id=0);

	void Step();
	void UpdateReward();
	void UpdateAdaptiveReward();
	void UpdateTerminalInfo();
	void Reset(bool RSI=true);
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
	Eigen::VectorXd GetState(bool dummy=false);
	void SetAction(const Eigen::VectorXd& action);
	double GetReward() {return mRewardParts[0]; }
	std::vector<double> GetRewardByParts() {return mRewardParts; }
	std::vector<std::string> GetRewardLabels() {return mRewardLabels; }
	const dart::simulation::WorldPtr& GetWorld() {return mWorld;}

	double GetTimeElapsed(){return this->mTimeElapsed;}
	double GetCurrentFrame(){return this->mCurrentFrame;}
	double GetCurrentLength() {return this->mCurrentFrame - this->mStartFrame; }
	double GetStartFrame(){ return this->mStartFrame; }

	const dart::dynamics::SkeletonPtr& GetSkeleton();

	void RescaleCharacter(double w0, double w1);
	void SaveDisplayedData(std::string directory);
	void SaveStats(std::string directory);
	void UpdateSigTorque();
	void UpdateGRF(std::vector<std::string> joints);
	std::vector<Eigen::VectorXd> GetGRF();
	void SaveStepInfo();
	Eigen::VectorXd GetPositions(int idx) { return this->mRecordPosition[idx]; }
	Eigen::Vector3d GetCOM(int idx) { return this->mRecordCOM[idx]; }
	Eigen::VectorXd GetVelocities(int idx) { return this->mRecordVelocity[idx]; }
	double GetTime(int idx) { return this->mRecordTime[idx]; }
	Eigen::VectorXd GetTargetPositions(int idx) { return this->mRecordTargetPosition[idx]; }
	Eigen::VectorXd GetBVHPositions(int idx) { return this->mRecordBVHPosition[idx]; }
	Eigen::VectorXd GetRewardPositions(int idx) { return this->mRecordRewardPosition[idx];}
	int GetRecordSize() { return this->mRecordPosition.size(); }
	std::pair<bool, bool> GetFootContact(int idx) { return this->mRecordFootContact[idx]; }
	std::tuple<double, double, double> GetRescaleParameter() { return mRescaleParameter; }
	
	void computeEnergyConservation();

	double ComputeLinearDifferenceFromEllipse();
	double ComputeAngularDifferenceFromEllipse(int idx);
	double ComputeAngularDifferenceFromCovarianceEllipse(int idx);
	std::vector<double> GetAdaptiveRefReward();

	std::vector<double> GetTrackingReward(Eigen::VectorXd position, Eigen::VectorXd position2, Eigen::VectorXd velocity, Eigen::VectorXd velocity2, std::vector<std::string> list, bool useVelocity);
	double GetTargetReward();
	std::vector<bool> GetContactInfo(Eigen::VectorXd pos);
	void GetNextPosition(Eigen::VectorXd cur, Eigen::VectorXd delta, Eigen::VectorXd& next);
	Eigen::VectorXd GetNewPositionFromAxisController(Eigen::VectorXd prev, double timestep, double phase);
	std::vector<double> GetAdaptiveIdxs();
protected:
	dart::simulation::WorldPtr mWorld;
	double w_p,w_v,w_com,w_ee,w_srl;
	double mStartFrame;
	double mCurrentFrame; // for discrete ref motion
	double mTimeElapsed;
	int mControlHz;
	int mSimulationHz;
	int mSimPerCon;
	double mCurrentFrameOnPhase;
	int nTotalSteps;
	bool isAdaptive;
	int id;

	Character* mCharacter;
	ReferenceManager* mReferenceManager;
	dart::dynamics::SkeletonPtr mGround;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	std::pair<bool, bool> mTargetContacts;

	Eigen::VectorXd mPDTargetPositions;
	Eigen::VectorXd mPDTargetVelocities;

	Eigen::VectorXd mRewardTargetPositions;

	Eigen::VectorXd mActions;

	Eigen::Vector3d mTargetCOMvelocity;
	double mAdaptiveCOM;
	double mAdaptiveStep;
	double sig_torque;
	double meanTargetReward;
	int mCount;

	std::vector<std::string> mInterestedBodies;
	std::vector<std::string> mRewardBodies;
	std::vector<std::string> mAdaptiveBodies;
	std::vector<std::string> mEndEffectors;
	std::vector<std::string> mRewardLabels;
	std::vector<double> mRewardParts;
	// for foot collision, left, right foot, ground
	std::unique_ptr<dart::collision::CollisionGroup> mCGEL, mCGER, mCGL, mCGR, mCGG, mCGHR, mCGHL; 

	std::vector<Eigen::VectorXd> mRecordPosition;
	std::vector<Eigen::VectorXd> mRecordVelocity;
	std::vector<Eigen::Vector3d> mRecordCOM;
	std::vector<Eigen::VectorXd> mRecordTargetPosition;
	std::vector<Eigen::VectorXd> mRecordBVHPosition;
	std::vector<Eigen::VectorXd> mRecordRewardPosition;

	std::vector<double> mRecordEnergy;
	std::vector<double> mRecordWork;
	std::vector<double> mRecordDCOM;
	std::vector<Eigen::VectorXd> mRecordTorque;
	std::vector<Eigen::VectorXd> mRecordWorkByJoints;
	std::vector<Eigen::VectorXd> mRecordTorqueByJoints;
	std::vector<std::pair<bool, bool>> mRecordFootContact;
	bool mIsTerminal;
	bool mIsNanAtTerminal;
	bool mRecord;
	std::tuple<bool, double, double> mDoubleStanceInfo;
	std::tuple<double, double, double> mRescaleParameter;
	std::vector<std::string> mGRFJoints;
	std::vector<double> mRecordTime;
	std::vector<double> mRecordDTime;
	std::vector<Eigen::VectorXd> mRecordFootConstraint;
	std::vector<Eigen::Vector6d> mRecordCOMVelocity;
	std::vector<Eigen::Vector3d> mRecordCOMPositionRef;
	std::pair<double, int> mInputVelocity;
	int mNumState, mNumAction;

	int terminationReason;

	std::vector<std::vector<Eigen::VectorXd>> GRFs;
	std::shared_ptr<dart::collision::DARTCollisionDetector> mGroundCollisionChecker;	

	Eigen::VectorXd mTorqueMean;
	Eigen::VectorXd mTorqueMin;
	Eigen::VectorXd mTorqueMax;

	Eigen::VectorXd mPrevPositions;
	Eigen::VectorXd mPrevTargetPositions;
	Eigen::VectorXd mTorqueSig;
	Eigen::VectorXd mMask;
	Eigen::VectorXd mControlFlag;

	Eigen::Vector3d mExtra;
	//target
	Eigen::Vector6d mHeadRoot;

	std::random_device mRD;
	std::mt19937 mMT;
	std::uniform_real_distribution<double> mDistribution;
	double mTarget;
	double mTarget2;
	double target_reward = 0;
	std::tuple<Eigen::VectorXd, double, double> mStartPosition;

};
}
#endif
