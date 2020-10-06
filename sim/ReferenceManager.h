#ifndef __DEEP_PHYSICS_REFERENCE_MANAGER_H__
#define __DEEP_PHYSICS_REFERENCE_MANAGER_H__

#include "Functions.h"
#include "Character.h"
#include "CharacterConfigurations.h"
#include "BVH.h"
#include "MultilevelSpline.h"
#include <tuple>
#include <mutex>

namespace DPhy
{
class Motion
{
public:
	Motion(Motion* m) {
		position = m->position;
		velocity = m->velocity;
		// rightContact = m->rightContact;
		// leftContact = m->leftContact;
	}
	Motion(Eigen::VectorXd pos) {
		position = pos;
	}
	Motion(Eigen::VectorXd pos, Eigen::VectorXd vel) {
		position = pos;
		velocity = vel;
	}
	void SetPosition(Eigen::VectorXd pos) { position = pos; }
	void SetVelocity(Eigen::VectorXd vel) { velocity = vel; }
	// void SetLeftContact(bool contact) { leftContact = contact; }
	// void SetRightContact(bool contact) { rightContact = contact; }

	Eigen::VectorXd GetPosition() { return position; }
	Eigen::VectorXd GetVelocity() { return velocity; }
	// bool GetLeftContact() { return leftContact; }
	// bool GetRightContact() { return rightContact; }

protected:
	Eigen::VectorXd position;
	Eigen::VectorXd velocity;
	// bool rightContact;
	// bool leftContact;
};
class ReferenceManager
{
public:
	ReferenceManager(Character* character=nullptr);
	void SaveAdaptiveMotion(std::string postfix="");
	void LoadAdaptiveMotion(std::vector<Eigen::VectorXd> cps);
	void LoadAdaptiveMotion(std::string postfix="");
	void LoadMotionFromBVH(std::string filename);
	void GenerateMotionsFromSinglePhase(int frames, bool blend, std::vector<Motion*>& p_phase, std::vector<Motion*>& p_gen);
	void RescaleMotion(double w);
	Motion* GetMotion(double t, bool adaptive=false);
	std::vector<Eigen::VectorXd> GetVelocityFromPositions(std::vector<Eigen::VectorXd> pos); 
	Eigen::VectorXd GetPosition(double t, bool adaptive=false);
	int GetPhaseLength() {return mPhaseLength; }
	void ComputeAxisDev();
	void ComputeAxisMean();
	Eigen::VectorXd GetAxisMean(double t);
	Eigen::VectorXd GetAxisDev(double t);
	double GetTimeStep(double t, bool adaptive);

	void SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, std::pair<double, double> rewards, Eigen::VectorXd parameters);
	void InitOptimization(int nslaves, std::string save_path);
	bool Optimize();
	void AddDisplacementToBVH(std::vector<Eigen::VectorXd> displacement, std::vector<Eigen::VectorXd>& position);
	void GetDisplacementWithBVH(std::vector<std::pair<Eigen::VectorXd, double>> position, std::vector<std::pair<Eigen::VectorXd, double>>& displacement);
	std::vector<std::pair<bool, Eigen::Vector3d>> GetContactInfo(Eigen::VectorXd pos);
	std::vector<double> GetContacts(double t);
	std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<double>> GetRegressionSamples();
	int GetDOF() {return mDOF; }
	int GetNumCPS() {return (mKnots.size()+3);}
	std::vector<double> GetKnots() {return mKnots;}
	void SetRefUpdateMode(bool on) { mRefUpdateMode = on; }
	Eigen::VectorXd GetTargetBase() {return mTargetBase; }
	Eigen::VectorXd GetTargetGoal() {return mTargetGoal; }
	Eigen::VectorXd GetTargetUnit() {return mTargetUnit; }
	Eigen::VectorXd GetTargetCurMean() {return mTargetCurMean; }
	bool UpgradeExternalTarget();
	void ReportEarlyTermination();
protected:
	Character* mCharacter;
	double mTimeStep;
	int mBlendingInterval;
	int mPhaseLength;
	std::vector<bool> mFootSliding;
	std::vector<Motion*> mMotions_raw;
	std::vector<Motion*> mMotions_phase;
	std::vector<Motion*> mMotions_phase_adaptive;
	std::vector<std::vector<bool>> mContacts;
	std::vector<Motion*> mMotions_gen;
	std::vector<Motion*> mMotions_gen_adaptive;
	std::vector<std::vector<Motion*>> mMotions_gen_temp;
	std::vector<double> mTimeStep_adaptive;

	std::vector<double> mIdxs;
	
	std::vector<Eigen::VectorXd> mAxis_BVH;
	std::vector<Eigen::VectorXd> mDev_BVH;

	//cps, target, similarity
	std::vector<std::tuple<std::pair<MultilevelSpline*, MultilevelSpline*>, std::pair<double, double>, double>> mSamples;
	
	//cps, parameter, quality
	std::vector<std::tuple<std::vector<Eigen::VectorXd>, Eigen::VectorXd, double>> mRegressionSamples;

	std::vector<Eigen::VectorXd> mPrevCps;
	std::vector<Eigen::VectorXd> mPrevCps_t;
	std::vector<Eigen::VectorXd> mDisplacement;
	std::vector<double> mKnots;
	std::vector<double> mKnots_t;
	std::vector<std::string> mInterestedBodies;
	std::vector<Eigen::VectorXd> mSampleTargets;
	double mSlaves;
	std::mutex mLock;
	std::mutex mLock_ET;

	bool mSaveTrajectory;
	std::string mPath;
	double mPrevRewardTrajectory;
	double mPrevRewardTarget;
	
	bool mRefUpdateMode;
	int mDOF;
	int nOp;
	Eigen::VectorXd mTargetBase;
	Eigen::VectorXd mTargetGoal;
	Eigen::VectorXd mTargetUnit;
	Eigen::VectorXd mTargetCurMean;
	
	std::vector<int> nRejectedSamples;
	double mMeanTrackingReward;
	int nET;
	int nT;
};
}

#endif