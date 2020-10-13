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
struct Param
{
	Eigen::VectorXd param;
	std::vector<Eigen::VectorXd> cps;
	double reward;
};
class ParamBin
{
public:
	ParamBin(Eigen::VectorXd i) { idx = i; }
	Eigen::VectorXd GetIdx(){ return idx; }
	void PutParam(Param p) { param.push_back(p); }
	int GetNumParams() { return param.size(); }
	std::vector<Param> GetParams() {return param;}
	void PutParams(std::vector<Param> ps) { param = ps;}
private:
	Eigen::VectorXd idx;
	std::vector<Param> param;
};
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
	
	bool Optimize();

	void SaveAdaptiveMotion();
	void LoadAdaptiveMotion();
	
	void TrainRegressionNetwork();
	p::list GetHindsightTuples();

	double GetPhaseLength();
	int GetDOF();
	
	void SetTargetParameters(np::ndarray np_array);

	bool NeedRefUpdate();
	void SetRefUpdateMode(bool t);
	p::list GetTargetBound();
	np::ndarray GetTargetBase();
	np::ndarray GetTargetUnit();
	np::ndarray GetTargetGoal();

	void AssignParamsToBins(bool limit=true);
	void RefreshTrainingData();
	void LoadParamBins();
private:
	std::vector<DPhy::Controller*> mSlaves;
	DPhy::ReferenceManager* mReferenceManager;
	int mNumSlaves;
	int mNumState;
	int mNumAction;
	bool isAdaptive;
	bool mNeedRefUpdate;
	bool isParametric;
	int mParamStack;
	int nTrainingData;
	int nCapacity;

	int nDim;
	Eigen::VectorXd mParamBase;
	Eigen::VectorXd mParamGoalIdx;
	//point, distance from zero point
	Eigen::VectorXd mParamUnit;

	std::vector<std::pair<Param, Eigen::VectorXd>> mParamNotAssigned;
	std::vector<ParamBin> mParamBins;
	
	p::object mRegression;

	std::string mPath;
};


#endif