#ifndef __DEEP_PHYSICS_H__
#define __DEEP_PHYSICS_H__
#include "Controller.h"
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
	
	SimEnv(int num_slaves, std::string motion, std::string mode);
	//For general properties
	int GetNumState();
	int GetNumAction();
	p::tuple GetDeformParameter();

	//For each slave
	void Step(int id);
	void Reset(int id,bool RSI);
	p::tuple IsNanAtTerminal(int id);
	void DeformCharacter(double w);

	np::ndarray GetState(int id);
	void SetAction(np::ndarray np_array,int id);
	double GetReward(int id);
	np::ndarray GetRewardByParts(int id);
	//For all slaves

	void Steps();
	void Resets(bool RSI);

	np::ndarray GetStates();
	void SetActions(np::ndarray np_array);
	np::ndarray GetRewards();
	np::ndarray GetRewardsByParts();

	void UpdateTarget(std::string directory);
private:
	std::vector<DPhy::Controller*> mSlaves;
	int mNumSlaves;
	int mNumState;
	int mNumAction;
};


#endif