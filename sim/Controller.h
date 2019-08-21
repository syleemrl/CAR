#ifndef __DEEP_PHYSICS_CONTROLLER_H__
#define __DEEP_PHYSICS_CONTROLLER_H__
#include "dart/dart.hpp"
#include "BVH.h"
#include "CharacterConfigurations.h"
#include "SkeletonBuilder.h"
#include "Functions.h"
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
Controller(std::string motion);

	void Step();
	void UpdateReward();
	void UpdateTerminalInfo();
	void Reset(bool RSI=true);
	void SetReference(std::string motion);
	bool FollowBvh();
	bool IsTerminalState() {return this->mIsTerminal; }
	bool IsNanAtTerminal() {return this->mIsNanAtTerminal;}
	bool IsTargetMet();

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

	double GetCurrentTime(){return this->mTimeElapsed;}
	double GetCurrentCount(){return this->mControlCount;}
	double GetCurrentLength() {return this->mControlCount - this->mStartCount; }
	double GetStartCount(){ return this->mStartCount; }
	std::string GetContactNodeName(int i);

	const dart::dynamics::SkeletonPtr& GetSkeleton();
	const dart::dynamics::SkeletonPtr& GetRefSkeleton();

	void DeformCharacter(double w);
	void SetNewTarget(double w);

	Eigen::VectorXd GetAdaptivePosition() {return mAdaptiveTargetPositions; };
protected:
	dart::simulation::WorldPtr mWorld;
	BVH* mBVH;
	double w_p,w_v,w_com,w_ee,w_goal;
	double mTimeElapsed;
	double mStartCount;
	int mControlCount; // for discrete ref motion
	int mControlHz;
	int mSimulationHz;
	int mSimPerCon;
	double mStep;
	
	Character* mCharacter;
	Character* mRefCharacter;
	dart::dynamics::SkeletonPtr mGround;

	double mTargetCOM, mTargetLf, mTargetRf;
	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mTargetContacts;

	Eigen::VectorXd mAdaptiveTargetPositions;
	Eigen::VectorXd mAdaptiveTargetVelocities;

	Eigen::VectorXd mPDTargetPositions;
	Eigen::VectorXd mPDTargetVelocities;

	Eigen::VectorXd mActions;


	std::vector<std::string> mInterestedBodies;
	std::vector<std::string> mRewardBodies;
	std::vector<std::string> mEndEffectors;

	std::vector<double> mRewardParts;
	// for foot collision, left, right foot, ground
	std::unique_ptr<dart::collision::CollisionGroup> mCGEL, mCGER, mCGL, mCGR, mCGG, mCGHR, mCGHL; 

	bool mIsTerminal;
	bool mIsNanAtTerminal;
	bool mTargetMet;

	int mNumState, mNumAction;

	int terminationReason;

	std::vector<Eigen::VectorXd> torques;

	std::shared_ptr<dart::collision::DARTCollisionDetector> mGroundCollisionChecker;	

};
}
#endif
