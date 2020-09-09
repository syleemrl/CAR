#ifndef __DEEP_PHYSICS_H__
#define __DEEP_PHYSICS_H__
#include "Controller.h"
// #include "SimpleController.h"
#include "ReferenceManager.h"
#include <vector>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <queue>
namespace DPhy
{
	class Controller;
}
namespace p = boost::python;
namespace np = boost::python::numpy;
class SimEnv
{
public:
	
	SimEnv(int num_slaves, std::string ref, std::string training_path, bool adaptive);
	//For general properties
	int GetNumState();
	int GetNumAction();

	//For each slave
	void Step(int id);
	void Reset(int id,bool RSI);
	p::tuple IsNanAtTerminal(int id);

	np::ndarray GetState(int id);
	void SetAction(np::ndarray np_array,int id);
	double GetReward(int id);
	np::ndarray GetRewardByParts(int id);
	//For all slaves

	void Steps();
	void Resets(bool RSI);

	np::ndarray GetStates();
	void SetActions(np::ndarray np_array);
	p::list GetRewardLabels();
	np::ndarray GetRewards();
	np::ndarray GetRewardsByParts();
	
	bool Optimize();

	void SaveAdaptiveMotion();
	void LoadAdaptiveMotion();
	
	void TrainRegressionNetwork();
	p::list GetHindsightTuples();

	double GetPhaseLength();
	int GetDOF();
	
	void SetTargetParameters(np::ndarray np_array);
	void SetRefUpdateMode(bool t);
	p::list GetTargetBound();
private:
	std::vector<DPhy::Controller*> mSlaves;
	DPhy::ReferenceManager* mReferenceManager;
	int mNumSlaves;
	int mNumState;
	int mNumAction;
	bool isAdaptive;

	double mTargetInterval;
	int nBins;
	std::pair<double, double> mMaxbound;
	std::vector<int> mTargetBin;
	bool mFlag_max;

	p::object mRegression;
};


#endif