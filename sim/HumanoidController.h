#ifndef __DEEP_PHYSICS_HUMANOID_CONTROLLER_H__
#define __DEEP_PHYSICS_HUMANOID_CONTROLLER_H__

#include "Controller.h"
#include "Functions.h"
#include "Humanoid.h"
#include "MotionGenerator.h"
#include "ReferenceManager.h"
#include "BVH.h"
#include "ThrowingBall.h"

namespace DPhy
{
class HumanoidController : public Controller
{
public:
	HumanoidController(bool use_trajectory=false, bool use_terminal = true, bool discrete_reference=false);

	virtual void Step(bool record=false) override;
	virtual void Reset(bool RSI=true) override;
	void FollowBvh();
	virtual void ResetWithTime(double time=0.0) override;
	virtual bool IsTerminalState() override;
	virtual bool IsNanAtTerminal(){return this->mIsNanAtTerminal;}
	virtual bool IsTimeEnd(){
		if(this->terminationReason == 8)
			return true;
		else
			return false;
	}


	virtual int GetNumState() override;
	virtual int GetNumAction() override;
	Eigen::VectorXd GetPositionState(const Eigen::VectorXd& pos);
	Eigen::VectorXd GetEndEffectorState(const Eigen::VectorXd& pos);
	Eigen::VectorXd GetEndEffectorStatePosAndVel(const Eigen::VectorXd& pv);

	bool CheckCollisionWithGround(std::string bodyName);
	virtual Eigen::VectorXd GetState() override;
	virtual void SetAction(const Eigen::VectorXd& action) override;
	virtual double GetReward() override;

	int GetHumanoidDof(){return this->mHumanoid->GetSkeleton()->getNumDofs();}
	void UpdateInitialState(Eigen::VectorXd mod);

	std::vector<double> GetRewardByParts();

	void UpdateReferenceDataForCurrentTime();
	Eigen::VectorXd GetTargetPositions();
	Eigen::VectorXd GetFuturePositions(double time, int index=-1);
	Eigen::VectorXd GetFuturePositions(int count, int index=-1);
	Eigen::VectorXd GetDecomposedPositions();

	virtual void Record() override;
	virtual void WriteRecords(const std::string& filename) override;
	virtual void WriteCompactRecords(const std::string& filename) override;

	// for goal
	double GetCurrentYRotation();
	double GetCurrentDirection();
	Eigen::Vector3d GetGoal(){return this->mGoal;}
	void SetGoal(const Eigen::Vector3d& goal){this->mGoal = goal;}
	void SetGoalTrajectory(const std::vector<Eigen::Vector3d>& goal_trajectory);
	void SaveGoalTrajectory();

	void SetReferenceToTarget(const Eigen::VectorXd& ref_cur, const Eigen::VectorXd& ref_next);
	void SetReferenceTrajectory(const Eigen::MatrixXd& trajectory);
    void SaveReferenceTrajectory();
	void FollowReference();
	void ApplyForce(std::string bodyname, Eigen::Vector3d force);

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

	void createNewBall();
	ThrowingBall* getThrowingBall();
	bool isUseBall(){return this->useBall;}
	
protected:
	MotionGenerator* mMotionGenerator;
	std::vector<ReferenceManager*> mReferenceManagers;
	int mCurrentReferenceManagerIndex;
	BVH* mBVH;
	double w_p,w_v,w_com,w_root_ori,w_root_av,w_ee,w_goal;
	double mTimeElapsed;
	int mControlCount; // for discrete ref motion
	double mStartTime;
	int mControlHz;
	int mSimulationHz;
	std::mt19937_64 mGenerator;
	
	Humanoid* mHumanoid;

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

	// goal setting : global position
	Eigen::Vector3d mGoal;

	std::vector<Eigen::Vector3d> mVelRecords, mGoalRecords;
	std::vector<double> mTimeRecords;
	std::vector<Eigen::VectorXd> mRefRecords, mModRecords;
	std::vector<Eigen::Vector2d> mFootRecords, mRefFootRecords;
	std::vector<Eigen::VectorXd> mBallRecords;

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

	bool useBall=true;
	bool mUseTerminal=true;
	ThrowingBall* mThrowingBall;

};
}

#endif