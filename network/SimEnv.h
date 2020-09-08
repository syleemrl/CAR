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
	
	Eigen::VectorXd PickTargetParameters(bool adaptive);
	void SetTargetParameters(Eigen::VectorXd tp);
	void SetSubTrainingMode(bool on);

private:
	std::vector<DPhy::Controller*> mSlaves;
//	std::vector<DPhy::SimpleController*> mSlaves;
	DPhy::ReferenceManager* mReferenceManager;
	int mNumSlaves;
	int mNumState;
	int mNumAction;
	bool isAdaptive;
	bool isSubtraining;

	p::object mRegression;

	std::vector<double> mTargetBins;

	std::vector<std::queue<double>> mTargetRewards;
	std::vector<double> mTargetMeanRewards;

};


#endif