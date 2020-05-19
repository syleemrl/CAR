#ifndef __DEEP_PHYSICS_REFERENCE_MANAGER_H__
#define __DEEP_PHYSICS_REFERENCE_MANAGER_H__

#include "Functions.h"
#include "Character.h"
#include "CharacterConfigurations.h"
#include "BVH.h"

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
	void SaveAdaptiveMotion();
	void LoadAdaptiveMotion();
	void LoadMotionFromBVH(std::string filename);
	void GenerateMotionsFromSinglePhase(int frames, bool blend, bool adaptive=false);
	void RescaleMotion(double w);
	void InitializeAdaptiveSettings(std::vector<double> idxs, double nslaves, std::string path = "", bool saveTrajectory=false);
	void SaveTuple(double time, Eigen::VectorXd position, int slave);
	void EndEpisode(int slave);
	void EndPhase(int slave);
	void SetTargetReward(double reward, int slave);
	void UpdateMotion();
	Motion* GetMotion(double t, bool adaptive=false);
	Eigen::VectorXd GetPosition(double t, bool adaptive=false);
	double GetTimeStep() {return mTimeStep; }
	int GetPhaseLength() {return mPhaseLength; }
	std::pair<bool, bool> CalculateContactInfo(Eigen::VectorXd p, Eigen::VectorXd v);
	void SaveEliteTrajectories(int slave);
protected:
	Character* mCharacter;
	double mTimeStep;
	int mBlendingInterval;
	int mPhaseLength;
	std::vector<Motion*> mMotions_raw;
	std::vector<Motion*> mMotions_phase;
	std::vector<Motion*> mMotions_phase_adaptive;
	std::vector<Motion*> mMotions_gen;
	std::vector<Motion*> mMotions_gen_adaptive;
	std::vector<double> mIdxs;
	
	//position, target
	std::vector<std::vector<Eigen::VectorXd>> mTuples;
	//time, target, position
	std::vector<std::vector<std::tuple<double, double, Eigen::VectorXd>>> mTuples_temp;
	std::vector<std::vector<Eigen::VectorXd>> mTuples_position;

	std::vector<Eigen::VectorXd> mPrevPosition;
	std::vector<double> mTargetReward;
	std::vector<Eigen::VectorXd> mAxis;

	double mSlaves;
	std::mutex mLock;
	bool mSaveTrajectory;
	std::string mPath;
};
}

#endif