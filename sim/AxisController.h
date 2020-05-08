#ifndef __DEEP_PHYSICS_AXISCONTROLLER_H__
#define __DEEP_PHYSICS_AXISCONTROLLER_H__
#include "dart/dart.hpp"
#include "Functions.h"
#include "ReferenceManager.h"

namespace DPhy
{
class AxisController
{
public:
	AxisController(int nslaves=4) { mslaves = nslaves; };

	void SetIdxs(std::vector<double> idxs);
	void Initialize(ReferenceManager* referenceManager);
	void SaveTuple(double time, double interval, Eigen::VectorXd position, int slave);
	void SetStartPosition(int slave);
	void EndPhase(int slave);
	void EndEpisode(int slave);
	void SetTargetReward(double reward, int slave);
	void UpdateAxis();
	void UpdateDev();
	void Save(std::string path);
	void Load(std::string path);
	Eigen::VectorXd GetMean(double time);
	Eigen::VectorXd GetDev(double time);

protected:
	std::vector<double> mIdxs;

	//target, position
	std::vector<std::vector<std::pair<double, Eigen::VectorXd>>> mTuples;
	//time, target, position
	std::vector<std::vector<std::tuple<double, double, Eigen::VectorXd>>> mTuples_temp;
	std::vector<Eigen::VectorXd> mPrevPosition;
	std::vector<double> mTargetReward;

	std::vector<Eigen::VectorXd> mAxis;
	std::vector<Eigen::VectorXd> mDev;

	double rewards;
	int mslaves;
	Eigen::VectorXd mStartPosition;
};
}
#endif
