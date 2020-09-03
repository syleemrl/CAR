#include "SimEnv.h"
#include <omp.h>
#include "dart/math/math.hpp"
#include "Functions.h"
#include <iostream>

SimEnv::
SimEnv(int num_slaves, std::string ref, std::string training_path, bool adaptive)
	:mNumSlaves(num_slaves)
{
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

	dart::math::seedRand();
	omp_set_num_threads(num_slaves);

	DPhy::Character* character = new DPhy::Character(path);
	mReferenceManager = new DPhy::ReferenceManager(character);
	mReferenceManager->LoadMotionFromBVH(ref);

	if(adaptive) {
		mReferenceManager->InitOptimization(num_slaves, training_path);
	}
	
	for(int i =0;i<num_slaves;i++)
	{
		mSlaves.push_back(new DPhy::Controller(mReferenceManager, adaptive, false, i));
	//	mSlaves.push_back(new DPhy::SimpleController());

	}
	

	mNumState = mSlaves[0]->GetNumState();
	mNumAction = mSlaves[0]->GetNumAction();
}
//For general properties
int
SimEnv::
GetNumState()
{
	return mNumState;
}
int
SimEnv::
GetNumAction()
{
	return mNumAction;
}

//For each slave
void 
SimEnv::
Step(int id)
{
	if(mSlaves[id]->IsTerminalState()){
		return;
	}
	mSlaves[id]->Step();
}
void 
SimEnv::
Reset(int id,bool RSI)
{
	mSlaves[id]->Reset(RSI);
}
p::tuple 
SimEnv::
IsNanAtTerminal(int id)
{
	bool t = mSlaves[id]->IsTerminalState();
	bool n = mSlaves[id]->IsNanAtTerminal();
	int start = mSlaves[id]->GetStartFrame();
	double e = mSlaves[id]->GetCurrentLength();
	double tt = mSlaves[id]->GetTimeElapsed();
	return p::make_tuple(t, n, start, e, tt);
}
np::ndarray
SimEnv::
GetState(int id)
{
	return DPhy::toNumPyArray(mSlaves[id]->GetState());
}
void 
SimEnv::
SetAction(np::ndarray np_array,int id)
{
	mSlaves[id]->SetAction(DPhy::toEigenVector(np_array,mNumAction));
}
p::list 
SimEnv::
GetRewardLabels()
{
	p::list l;
	std::vector<std::string> sl = mSlaves[0]->GetRewardLabels();
	for(int i =0 ; i <sl.size(); i++) l.append(sl[i]);
	return l;
}
double 
SimEnv::
GetReward(int id)
{
	return mSlaves[id]->GetReward();
}
np::ndarray
SimEnv::
GetRewardByParts(int id)
{
	std::vector<double> ret;
	if(dynamic_cast<DPhy::Controller*>(mSlaves[id])!=nullptr){
		ret = dynamic_cast<DPhy::Controller*>(mSlaves[id])->GetRewardByParts();
	}
	return DPhy::toNumPyArray(ret);
}
void
SimEnv::
Steps()
{
	if( mNumSlaves == 1){
		this->Step(0);
	}
	else{
#pragma omp parallel for
		for (int id = 0; id < mNumSlaves; ++id)
		{
			this->Step(id);
		}
	}
}
void
SimEnv::
Resets(bool RSI)
{
	for (int id = 0; id < mNumSlaves; ++id)
	{
		this->Reset(id,RSI);
	}
}
np::ndarray
SimEnv::
GetStates()
{
	Eigen::MatrixXd states(mNumSlaves,mNumState);

	for (int id = 0; id < mNumSlaves; ++id)
	{
		states.row(id) = mSlaves[id]->GetState().transpose();
	}
	return DPhy::toNumPyArray(states);
}
void
SimEnv::
SetActions(np::ndarray np_array)
{
	Eigen::MatrixXd action = DPhy::toEigenMatrix(np_array,mNumSlaves,mNumAction);

	for (int id = 0; id < mNumSlaves; ++id)
	{
		mSlaves[id]->SetAction(action.row(id).transpose());
	}
}
np::ndarray
SimEnv::
GetRewards()
{
	std::vector<float> rewards(mNumSlaves);
	for (int id = 0; id < mNumSlaves; ++id)
	{
		rewards[id] = this->GetReward(id);
	}

	return DPhy::toNumPyArray(rewards);
}
np::ndarray
SimEnv::
GetRewardsByParts()
{
	std::vector<std::vector<double>> rewards(mNumSlaves);
	for (int id = 0; id < mNumSlaves; ++id)
	{
		if(dynamic_cast<DPhy::Controller*>(mSlaves[id])!=nullptr){
			rewards[id] = dynamic_cast<DPhy::Controller*>(mSlaves[id])->GetRewardByParts();
		}
	}

	return DPhy::toNumPyArray(rewards);
}
bool
SimEnv::
Optimize()
{
	bool t = mReferenceManager->Optimize();
	return t;
}
p::list
SimEnv::
GetRegressionSamples()
{
	std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> x_y = mReferenceManager->GetRegressionSamples();
	np::ndarray x = DPhy::toNumPyArray(x_y.first);
	np::ndarray y = DPhy::toNumPyArray(x_y.second);

	p::list l;
	l.append(x);
	l.append(y);

	return l;
}
void
SimEnv::
LoadAdaptiveMotion()
{
	mReferenceManager->LoadAdaptiveMotion();
}
double 
SimEnv::
GetPhaseLength()
{
	return mReferenceManager->GetPhaseLength();
}
using namespace boost::python;

BOOST_PYTHON_MODULE(simEnv)
{
	Py_Initialize();
	np::initialize();

	class_<SimEnv>("Env",init<int, std::string, std::string, bool>())
		.def("GetPhaseLength",&SimEnv::GetPhaseLength)
		.def("GetNumState",&SimEnv::GetNumState)
		.def("GetNumAction",&SimEnv::GetNumAction)
		.def("Step",&SimEnv::Step)
		.def("Reset",&SimEnv::Reset)
		.def("GetState",&SimEnv::GetState)
		.def("SetAction",&SimEnv::SetAction)
		.def("GetRewardLabels",&SimEnv::GetRewardLabels)
		.def("GetReward",&SimEnv::GetReward)
		.def("GetRewardByParts",&SimEnv::GetRewardByParts)
		.def("Steps",&SimEnv::Steps)
		.def("Resets",&SimEnv::Resets)
		.def("IsNanAtTerminal",&SimEnv::IsNanAtTerminal)
		.def("GetStates",&SimEnv::GetStates)
		.def("SetActions",&SimEnv::SetActions)
		.def("GetRewards",&SimEnv::GetRewards)
		.def("GetRegressionSamples",&SimEnv::GetRegressionSamples)
		.def("Optimize",&SimEnv::Optimize)
		.def("LoadAdaptiveMotion",&SimEnv::LoadAdaptiveMotion)
		.def("GetRewardsByParts",&SimEnv::GetRewardsByParts);
}