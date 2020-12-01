#ifndef __DEEP_PHYSICS_H__
#define __DEEP_PHYSICS_H__
#include "Controller.h"
// #include "SimpleController.h"
#include "ReferenceManager.h"
#include "RegressionMemory.h"
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
	
	SimEnv(int num_slaves, std::string ref, std::string training_path, bool adaptive, bool parametric);
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
	np::ndarray GetParamGoal();
	np::ndarray UniformSample(bool visited);

	void LoadAdaptiveMotion();
	void TrainRegressionNetwork();

	double GetPhaseLength();
	int GetDOF();
	
	void SetGoalParameters(np::ndarray np_array, bool mem_only);
	void UpdateReference();
	void SaveParamSpace(int n);
	void SaveParamSpaceLog(int n);

	double GetVisitedRatio();
	double GetDensity(np::ndarray np_array);
	void UpdateParamState();
	p::list GetParamSpaceSummary();
	p::list UniformSampleWithNearestParams();
	p::list GetNearestParams(np::ndarray np_array);
	double GetProgressGoal();

private:
	std::vector<DPhy::Controller*> mSlaves;
	DPhy::ReferenceManager* mReferenceManager;
	DPhy::RegressionMemory* mRegressionMemory;

	int mExUpdate;
	int mNumSlaves;
	int mNumState;
	int mNumAction;
	
	p::object mRegression;

	std::string mPath;
};


#endif