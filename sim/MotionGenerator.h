#ifndef __DEEP_PHYSICS_MOTION_GENERATOR_H__
#define __DEEP_PHYSICS_MOTION_GENERATOR_H__
#include "BVH.h"
#include "Character.h"
#include "dart/dart.hpp"

namespace DPhy
{
class MotionGenerator
{
public:
	MotionGenerator(Character* character=nullptr);

	void Initialize();

	// get motion(joint positions) at time
	std::pair<Eigen::VectorXd, Eigen::VectorXd> getMotion(double time);

	// add bvh file to motion generator
	void addBVH(std::string motionfilename);

	// add bvh file to motion generator
	void addBVHs(std::string motionfilename);

	// set next motion clip
	void setNext(int index);

	// set character
	void setCharacter(Character* character){this->mCharacter = character;}

	// get max time of current motion
	double getMaxTime();

	// apply offset to positions
	Eigen::VectorXd applyOffset(Eigen::VectorXd p, double time_in_motion);

	// TODO
	// Motion blending
	// COM, Orientation
	// Motion transition implementation
	// Recording, parsing


protected:
	std::vector<BVH*> mBVHs;
	Character* mCharacter;

	int mCurrentMotion, mNextMotion;
	double mMotionStartTime, mMotionDuration;
	Eigen::VectorXd mPositionDifferenceWithLastMotion;
	Eigen::VectorXd mFirstPositions;

	Eigen::Vector3d mCOMOffsetGlobal, mCOMOffsetCurrent;
	Eigen::Quaterniond mOrientationOffsetGlobal, mOrientationOffsetCurrent;
};
}

#endif