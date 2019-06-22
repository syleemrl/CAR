#ifndef __DEEP_PHYSICS_CONTROLLER_H__
#define __DEEP_PHYSICS_CONTROLLER_H__
#include "dart/dart.hpp"
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
	void Reset(bool RSI=true);
	void SetReference(std::string motion);
	void FollowBvh();
	bool IsTerminalState();
	bool IsNanAtTerminal() {return this->mIsNanAtTerminal;}
	bool IsTimeEnd(){
		if(this->terminationReason == 8)
			return true;
		else
			return false;
	}
	int GetNumState();
	int GetNumAction();
	Eigen::VectorXd GetEndEffectorStatePosAndVel(const Eigen::VectorXd& pv);

	bool CheckCollisionWithGround(std::string bodyName);
	Eigen::VectorXd GetState();
	void SetAction(const Eigen::VectorXd& action);
	double GetReward();
	std::vector<double> GetRewardByParts();
	const dart::simulation::WorldPtr& GetWorld() {return mWorld;}

	double GetCurrentTime(){return this->mTimeElapsed;}
	double GetCurrentCount(){return this->mControlCount;}

	
protected:
	dart::simulation::WorldPtr mWorld;
	BVH* mBVH;
	double w_p,w_v,w_com,w_ee;
	double mTimeElapsed;
	int mControlCount; // for discrete ref motion
	int mControlHz;
	int mSimulationHz;
	int mSimPerCon;

	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;

	Eigen::VectorXd mModifiedTargetPositions;
	Eigen::VectorXd mModifiedTargetVelocities;

	Eigen::VectorXd mActions;

	Eigen::Vector2d mFoot, mRefFoot;

	std::vector<std::string> mInterestedBodies;
	std::vector<std::string> mRewardBodies;
	std::vector<std::string> mRewardUpperBodies;

	std::vector<std::string> mEndEffectors, mRewardEndEffectors;

	// for foot collision, left, right foot, ground
	std::unique_ptr<dart::collision::CollisionGroup> mCGEL, mCGER, mCGL, mCGR, mCGG; 

	bool mIsTerminal;
	bool mIsNanAtTerminal;

	int mNumState, mNumAction;

	int terminationReason;

	std::vector<Eigen::VectorXd> torques;

	std::shared_ptr<dart::collision::DARTCollisionDetector> mGroundCollisionChecker;	

};
}
#endif
