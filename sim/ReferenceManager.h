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
	double GetTimeStep() {return mTimeStep; }
	int GetPhaseLength() {return mPhaseLength; }
	void ComputeAxisDev();
	void ComputeAxisMean();
	Eigen::VectorXd GetAxisMean(double t);
	Eigen::VectorXd GetAxisDev(double t);

	void SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, std::pair<double, double> rewards, Eigen::VectorXd parameters);
	void InitOptimization(int nslaves, std::string save_path);
	bool Optimize();
	void AddDisplacementToBVH(std::vector<Eigen::VectorXd> displacement, std::vector<Eigen::VectorXd>& position);
	void GetDisplacementWithBVH(std::vector<std::pair<Eigen::VectorXd, double>> position, std::vector<std::pair<Eigen::VectorXd, double>>& displacement);
	std::vector<std::pair<bool, Eigen::Vector3d>> GetContactInfo(Eigen::VectorXd pos);
	std::vector<double> GetContacts(double t);
	std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> GetRegressionSamples();
	int GetDOF() {return mDOF; }
	int GetNumCPS() {return (mKnots.size()+3);}
	std::vector<double> GetKnots() {return mKnots;}
	void SetRefUpdateMode(bool on) { mRefUpdateMode = on; }
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

	std::vector<double> mIdxs;
	
	std::vector<Eigen::VectorXd> mAxis_BVH;
	std::vector<Eigen::VectorXd> mDev_BVH;

	//cps, target, similarity
	std::vector<std::tuple<MultilevelSpline*, std::pair<double, double>, double>> mSamples;
	
	//cps, parameter
	std::vector<std::pair<std::vector<Eigen::VectorXd>, Eigen::VectorXd>> mRegressionSamples;

	std::vector<Eigen::VectorXd> mPrevCps;
	std::vector<Eigen::VectorXd> mDisplacement;
	std::vector<double> mKnots;
	std::vector<std::string> mInterestedBodies;

	double mSlaves;
	std::mutex mLock;
	bool mSaveTrajectory;
	std::string mPath;
	double mPrevRewardTrajectory;
	double mPrevRewardTarget;
	bool mRefUpdateMode;
	int mDOF;
	int nOp;

	std::vector<int> nRejectedSamples;
};
}

#endif