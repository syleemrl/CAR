#ifndef __DEEP_PHYSICS_H__
#define __DEEP_PHYSICS_H__
#include "Controller.h"
#include "ReferenceManager.h"
#include <vector>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace DPhy
{
	class Controller;
}
namespace p = boost::python;
namespace np = boost::python::numpy;
class SimEnv
{
public:
	
	SimEnv(int num_slaves, std::string motion_file);
	//For general properties
	int GetNumState();
	int GetNumAction();

	//For each slave
	void Step(int id,bool record);
	void Reset(int id,bool RSI);
	bool IsTerminalState(int id);
	p::tuple IsNanAtTerminal(int id);

	np::ndarray GetState(int id);
	np::ndarray GetDecomposedPositions(int id);
	void SetAction(np::ndarray np_array,int id);
	double GetReward(int id);
	np::ndarray GetRewardByParts(int id);
	//For all slaves

	void Steps(bool record);
	void Resets(bool RSI);
	np::ndarray IsTerminalStates();

	np::ndarray GetStates();
	void SetActions(np::ndarray np_array);
	np::ndarray GetRewards();
	np::ndarray GetRewardsByParts();

private:
	std::vector<DPhy::Controller*> mSlaves;
	Dphy::ReferenceManager* mReference;
	int mNumSlaves;
	int mNumState;
	int mNumAction;
};


#endif