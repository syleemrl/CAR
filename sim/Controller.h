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
Controller();

	void Step(bool record=false);
	void Reset(bool RSI=true);
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
	Eigen::VectorXd GetPositionState(const Eigen::VectorXd& pos);
	Eigen::VectorXd GetEndEffectorState(const Eigen::VectorXd& pos);
	Eigen::VectorXd GetEndEffectorStatePosAndVel(const Eigen::VectorXd& pv);

	bool CheckCollisionWithGround(std::string bodyName);
	Eigen::VectorXd GetState();
	void SetAction(const Eigen::VectorXd& action);
	double GetReward();

	int GetHumanoidDof(){return this->mHumanoid->GetSkeleton()->getNumDofs();}
	void UpdateInitialState(Eigen::VectorXd mod);

	std::vector<double> GetRewardByParts();

	Eigen::VectorXd GetTargetPositions();
	Eigen::VectorXd GetDecomposedPositions();

	void AddReference(const Eigen::VectorXd& ref, int index=-1);
	Eigen::VectorXd GetReference(int index){
		this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getPositions(index);
	}
	void ClearReferenceManager(){
		for(int i = 0; i < REFERENCE_MANAGER_COUNT; i++)
			this->mReferenceManagers[i]->clear();
	}
	double GetCurrentTime(){return this->mTimeElapsed;}
	double GetCurrentCount(){return this->mControlCount;}
	double GetMaxTime(){
		return this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getMaxTime() - FUTURE_TIME;
	}
	int GetMaxCount(){
		return this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getMaxCount() - FUTURE_COUNT;
	}
	bool IsReferenceEnough(){
		if(this->mUseTrajectory){
			std::cout << "HumanoidController.h : IsReferenceEnough is called in not recursive simulation" << std::endl;
			return false;
		}
		else{
			if(this->mUseDiscreteReference)
				return this->GetMaxCount() >= 0;
			else
				return this->GetMaxTime() >= 0;
		}
	}

	void SetNextMotion(int index){
		this->mMotionGenerator->setNext(index);
	}

	Humanoid* getHumanoid(){ return mHumanoid; }
	double getControlHz(){ return mControlHz; }

	Eigen::VectorXd getLastReferenceMotion(){
	    return mReferenceManagers[this->mCurrentReferenceManagerIndex]->getLastReferenceMotion();
	}

	// DEBUG
	void ComputeRootCOMDiff();
	void GetRootCOMDiff();

	void UpdateMax();
	void UpdateMin();
	
protected:
	dart::simulation::WorldPtr mWorld;
	ReferenceManager* mReferenceManager;
	int mCurrentReferenceManagerIndex;
	BVH* mBVH;
	double w_p,w_v,w_com,w_root_ori,w_root_av,w_ee,w_goal;
	double mTimeElapsed;
	int mControlCount; // for discrete ref motion
	double mStartTime;
	int mControlHz;
	int mSimulationHz;
	std::mt19937_64 mGenerator;
	
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;

	Eigen::VectorXd mModifiedTargetPositions;
	Eigen::VectorXd mModifiedTargetVelocities;

	Eigen::VectorXd mMaxJoint, mMinJoint;
	Eigen::VectorXd mPositionUpperLimits, mPositionLowerLimits;
	Eigen::VectorXd mActionRange;

	Eigen::VectorXd mActions;

	Eigen::Vector2d mFoot, mRefFoot;

	std::vector<std::string> mInterestedBodies;
	std::vector<std::string> mRewardBodies;
	std::vector<std::string> mRewardUpperBodies;

	std::vector<std::string> mEndEffectors, mRewardEndEffectors;

	std::vector<std::string> mHumanoidRevolutedBodiesR,mHumanoidRevolutedBodiesL;

	std::string mReferenceMotionFilename;

	// for foot collision, left, right foot, ground
	std::unique_ptr<dart::collision::CollisionGroup> mCGEL, mCGER, mCGL, mCGR, mCGG; 

	// DEBUG
	Eigen::Vector3d mRootCOMAtTerminal;
	Eigen::Vector3d mRootCOMAtTerminalRef;
	bool mIsTerminal;
	bool mIsNanAtTerminal;
	bool mUseTrajectory;
	bool mUseDiscreteReference;
	bool mUpdated;

	int mNumState, mNumAction;

	int terminationReason;

	std::vector<Eigen::VectorXd> torques;

	std::shared_ptr<dart::collision::DARTCollisionDetector> mGroundCollisionChecker;

	bool mUseTerminal=true;
	

};
}
#endif
