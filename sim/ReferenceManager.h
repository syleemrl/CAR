#ifndef __DEEP_PHYSICS_REFERENCE_MANAGER_H__
#define __DEEP_PHYSICS_REFERENCE_MANAGER_H__

#include "Functions.h"
#include "Character.h"
#include "CharacterConfigurations.h"
#include <tuple>


namespace DPhy
{
class ReferenceManager
{
public:
	ReferenceManager(Character* character=nullptr, double ctrl_hz = 50);

	double getMaxTime(){
		return (this->mNumTotalFrame-1)*(1.0/this->mMotionHz)-1.0/this->mControlHz;
	}
	int getMaxCount(){
		return this->mNumTotalFrame-2;
	}
	double getTimeStep();
	double getControlHz() const;

	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion();

	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion(double time);
	Eigen::VectorXd getPositions(double time);
	Eigen::VectorXd getPositionsAndVelocities(double time);
	Eigen::VectorXd getOriginPositions(double time);
	Eigen::Vector2d getFootContacts(double time);

	std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getMotion(int count);
	Eigen::VectorXd getPositions(int count);
	Eigen::VectorXd getPositionsAndVelocities(int count);
	Eigen::VectorXd getOriginPositions(int count);
	Eigen::Vector2d getFootContacts(int count);
	std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> getFootIKConstraints(bool isRight, bool isPull);

	void setTargetMotion(const Eigen::VectorXd& current_target, const Eigen::VectorXd& next_target);
	void setTrajectory(const Eigen::MatrixXd& trajectory);

	void setGoal(const Eigen::Vector3d& goal);
	void setGoalTrajectory(const std::vector<Eigen::Vector3d>& goal_trajectory);

	Eigen::Vector3d getGoal();
	Eigen::Vector3d getGoal(double time);
	Eigen::Vector3d getGoal(int count);

	Eigen::Quaterniond convertRNNMotion(const Eigen::VectorXd& ref, std::string bodyName, int index_in_input, const Eigen::Quaterniond& offset);
	void setRNNMotion(Eigen::VectorXd& converted_motion, const Eigen::Vector3d& rnn_motion, std::string bodyName);
	void setRNNMotion(Eigen::VectorXd& converted_motion, const Eigen::Quaterniond& rnn_motion, std::string bodyName);
	void convertAndSetRNNMotion(Eigen::VectorXd& converted_motion, const Eigen::VectorXd& ref, std::string bodyName, int index_in_input, const Eigen::Quaterniond& offset);
	Eigen::VectorXd convertRNNMotion(const Eigen::VectorXd& ref);

	void clear();
	void addPosition(const Eigen::VectorXd& ref_pos);

	/// HS) saving converted trajectory.
	void saveReferenceTrajectory(std::string filename);

	/// HS) loading converted trajectory.
	void saveGoalTrajectory(std::string filename);

	/// HS) return last reference motion
	Eigen::VectorXd getLastReferenceMotion(){
	    return mReferenceTrajectory[this->mNumTotalFrame-1];
	}

protected:
	int mMotionHz;
	int mNumTotalFrame;
	double mControlHz;

	Character* mCharacter;

	std::vector<Eigen::VectorXd> mReferenceTrajectory;
	std::vector<Eigen::VectorXd> mReferenceTrajectoryOrigin;
	std::vector<Eigen::Vector3d> mGoalTrajectory;
	std::vector<Eigen::Vector2d> mFootContactTrajectory;

	Eigen::VectorXd mCurrentReferencePosition, mNextReferencePosition, mCurrentReferenceVelocity;
	Eigen::Vector3d mGoal;
};
}

#endif