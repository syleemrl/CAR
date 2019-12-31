#ifndef __DEEP_PHYSICS_REFERENCE_MANAGER_H__
#define __DEEP_PHYSICS_REFERENCE_MANAGER_H__

#include "Functions.h"
#include "Character.h"
#include "CharacterConfigurations.h"
#include "BVH.h"

#include <tuple>


namespace DPhy
{
class Motion
{
public:
	Motion(Motion* m) {
		position = m->position;
		velocity = m->velocity;
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
	void LoadMotionFromBVH(std::string filename);
	void LoadMotionFromTrainedData(std::string filename);
	void RescaleMotion(double w);
	Motion* GetMotion(double t);
	double GetTimeStep() {return mTimeStep; }
	int GetPhaseLength() {return mPhaseLength; }
	double GetAvgWork() {return mAvgWork; }
	Eigen::VectorXd GetForce(int idx) {return mTorques_phase[idx]; }
	double GetWork(int idx) {return mWorks_phase[idx]; }

protected:
	Character* mCharacter;
	double mTimeStep;
	double mAvgWork;
	int mPhaseLength;
	std::vector<Motion*> mMotions_phase;
	std::vector<double> mWorks_phase;
	std::vector<Eigen::VectorXd> mTorques_phase;
	std::vector<Motion*> mMotions;
};
}

#endif