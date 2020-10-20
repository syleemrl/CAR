#ifndef __DEEP_PHYSICS_REFERENCE_MANAGER_H__
#define __DEEP_PHYSICS_REFERENCE_MANAGER_H__

#include "Functions.h"
#include "Character.h"
#include "CharacterConfigurations.h"
#include "BVH.h"
#include "MultilevelSpline.h"
#include "RegressionMemory.h"
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

	Eigen::VectorXd GetPosition() { return position; }
	Eigen::VectorXd GetVelocity() { return velocity; }

protected:
	Eigen::VectorXd position;
	Eigen::VectorXd velocity;

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
	Motion* GetMotion(double t, bool adaptive=false);
	std::vector<Eigen::VectorXd> GetVelocityFromPositions(std::vector<Eigen::VectorXd> pos); 
	Eigen::VectorXd GetPosition(double t, bool adaptive=false);
	int GetPhaseLength() {return mPhaseLength; }
	double GetTimeStep(double t, bool adaptive);

	void SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, std::pair<double, double> rewards, Eigen::VectorXd parameters);
	void InitOptimization(int nslaves, std::string save_path, bool isParametric=false);
	bool Optimize();
	void AddDisplacementToBVH(std::vector<Eigen::VectorXd> displacement, std::vector<Eigen::VectorXd>& position);
	void GetDisplacementWithBVH(std::vector<std::pair<Eigen::VectorXd, double>> position, std::vector<std::pair<Eigen::VectorXd, double>>& displacement);
	std::vector<std::pair<bool, Eigen::Vector3d>> GetContactInfo(Eigen::VectorXd pos);
	std::vector<double> GetContacts(double t);
	int GetDOF() {return mDOF; }
	int GetNumCPS() {return (mKnots.size()+3);}
	std::vector<double> GetKnots() {return mKnots;}
	void SetExplorationMode(bool on) { mExplorationMode = on; }
	Eigen::VectorXd GetParamGoal() {return mParamGoal; }
	Eigen::VectorXd GetParamCur() {return mParamCur; }
	std::pair<Eigen::VectorXd, Eigen::VectorXd> GetParamRange() {return std::pair<Eigen::VectorXd, Eigen::VectorXd>(mParamBase, mParamEnd); }

	void SetParamGoal(Eigen::VectorXd g) { mParamGoal = g; }
	void ResetOptimizationParameters();
	bool UpdateParamManually();
	bool CheckExplorationProgress();
	void ReportEarlyTermination();
	void SetRegressionMemory(RegressionMemory* r) {mRegressionMemory = r; }
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
	
	std::vector<Eigen::VectorXd> mAxis_BVH;
	std::vector<Eigen::VectorXd> mDev_BVH;

	//cps, target, similarity
	std::vector<std::tuple<std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>, 
						   std::pair<double, double>, 
						   double>> mSamples;
	
	std::vector<Eigen::VectorXd> mPrevCps;
	std::vector<Eigen::VectorXd> mPrevCps_t;
	std::vector<Eigen::VectorXd> mDisplacement;
	std::vector<double> mKnots;
	std::vector<double> mKnots_t;
	std::vector<std::string> mInterestedBodies;
	std::vector<Eigen::VectorXd> mSampleParams;

	double mSlaves;
	std::mutex mLock;
	std::mutex mLock_ET;

	bool mSaveTrajectory;
	std::string mPath;
	double mPrevRewardTrajectory;
	double mPrevRewardParam;
	
	bool mExplorationMode;
	bool isParametric;
	int mDOF;
	int nOp;

	Eigen::VectorXd mParamGoal;
	Eigen::VectorXd mParamCur;
	Eigen::VectorXd mParamBase;
	Eigen::VectorXd mParamEnd;

	RegressionMemory* mRegressionMemory;
	
	double mMeanTrackingReward;
	double mMeanParamReward;
	double mPrevMeanParamReward;

	double mThresholdTracking;
	double mThresholdSurvival;
	int mThresholdProgress;

	int nET;
	int nT;
	int nProgress;
};
}

#endif