#include "SimEnv.h"
#include <omp.h>
#include "dart/math/math.hpp"
#include "Functions.h"
#include <iostream>
SimEnv::
SimEnv(int num_slaves, std::string ref, std::string training_path, bool adaptive, bool parametric)
	:mNumSlaves(num_slaves)
{
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
	mPath = training_path;

	dart::math::seedRand();
	omp_set_num_threads(num_slaves);

	DPhy::Character* character = new DPhy::Character(path);
	mReferenceManager = new DPhy::ReferenceManager(character);
	mReferenceManager->LoadMotionFromBVH(ref);
	
	if(adaptive) {
		mRegressionMemory = new DPhy::RegressionMemory();
		mReferenceManager->SetRegressionMemory(mRegressionMemory);

		mReferenceManager->InitOptimization(num_slaves, training_path, adaptive);
		mReferenceManager->LoadAdaptiveMotion("");

		mRegressionMemory->LoadParamSpace(mPath + "param_space");
		
	} else {
		mReferenceManager->InitOptimization(num_slaves, "");
	}

	if(adaptive) {
		Py_Initialize();
		np::initialize();
		try{
			p::object regression = p::import("regression");
			this->mRegression = regression.attr("Regression")();
			this->mRegression.attr("initTrain")(training_path, mRegressionMemory->GetDim() + 1, mReferenceManager->GetDOF() + 1);
		}
		catch (const p::error_already_set&)
		{
			PyErr_Print();
		}
	}
	
	for(int i =0;i<num_slaves;i++)
	{
		mSlaves.push_back(new DPhy::Controller(mReferenceManager, adaptive, parametric, false, i));
		if(adaptive) {
			Eigen::VectorXd tp = mReferenceManager->GetParamGoal();
			mSlaves[i]->SetGoalParameters(tp);
		}
	}
	
	mNumState = mSlaves[0]->GetNumState();
	mNumAction = mSlaves[0]->GetNumAction();
	mExUpdate = 0;
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
	int term = mSlaves[id]->GetTerminationReason();
	return p::make_tuple(t, n, start, e, tt, term);
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
void 
SimEnv::
TrainRegressionNetwork()
{
	std::tuple<std::vector<Eigen::VectorXd>, 
			   std::vector<Eigen::VectorXd>, 
			   std::vector<double>> x_y_z = mRegressionMemory->GetTrainingData();
	if(std::get<0>(x_y_z).size() == 0)
		return;

	np::ndarray x = DPhy::toNumPyArray(std::get<0>(x_y_z));
	np::ndarray y = DPhy::toNumPyArray(std::get<1>(x_y_z));

	p::list l;
	l.append(x);
	l.append(y);

	this->mRegression.attr("setRegressionData")(l);
	this->mRegression.attr("train")();

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
int
SimEnv::
GetDOF()
{
	return mReferenceManager->GetDOF();
}
void 
SimEnv::
UpdateReference() {
	Eigen::VectorXd tp = mReferenceManager->GetParamGoal();		
		
	std::vector<Eigen::VectorXd> cps = mRegressionMemory->GetCPSFromNearestParams(tp);
	mReferenceManager->LoadAdaptiveMotion(cps);

	mReferenceManager->SaveAdaptiveMotion("ref_"+std::to_string(mExUpdate));
	mExUpdate += 1;

}
void 
SimEnv::
SetGoalParameters(np::ndarray np_array, bool mem_only) {

	int dim = mRegressionMemory->GetDim();
	Eigen::VectorXd tp = DPhy::toEigenVector(np_array, dim);
	Eigen::VectorXd tp_normalized = mRegressionMemory->Normalize(tp);
	int dof = mReferenceManager->GetDOF() + 1;
	int dof_input = 1 + mRegressionMemory->GetDim();
	std::vector<Eigen::VectorXd> cps;
	if(mem_only) {
		cps = mRegressionMemory->GetCPSFromNearestParams(tp);
		mReferenceManager->LoadAdaptiveMotion(cps);
	} else {
		// for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
		// 	Eigen::VectorXd input(dof_input);
		// 	input << j, tp_normalized;
		// 	p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
		// 	np::ndarray na = np::from_object(a);
		// 	cps.push_back(DPhy::toEigenVector(na, dof));
		// }
		// mReferenceManager->SetCPSreg(cps);
		// cps = mRegressionMemory->GetCPSFromNearestParams(tp);
		// mReferenceManager->SetCPSexp(cps);
		// mReferenceManager->SelectReference();
		cps = mRegressionMemory->GetCPSFromNearestParams(tp);
		mReferenceManager->LoadAdaptiveMotion(cps);
	}

	for(int id = 0; id < mNumSlaves; ++id) {
		mSlaves[id]->SetGoalParameters(tp);
	}
}
np::ndarray 
SimEnv::
GetParamGoal() {
	return DPhy::toNumPyArray(mReferenceManager->GetParamGoal());
}
np::ndarray 
SimEnv::
UniformSample(bool visited) {
	std::pair<Eigen::VectorXd , bool> pair = mRegressionMemory->UniformSample(visited);
	if(!pair.second) {
		std::cout << "exploration done" << std::endl;
	}
	return DPhy::toNumPyArray(pair.first);
}
void
SimEnv::
SaveParamSpace(int n) {
	if(n != -1) {
		mRegressionMemory->SaveParamSpace(mPath + "param_space" + std::to_string(n));
	} else {
		mRegressionMemory->SaveParamSpace(mPath + "param_space");
	}
}
void
SimEnv::
SaveParamSpaceLog(int n) {
	mRegressionMemory->SaveLog(mPath + "log");

}
void
SimEnv::
UpdateParamState() {
	mRegressionMemory->UpdateParamState();
}
double
SimEnv::
GetVisitedRatio() {
	return mRegressionMemory->GetVisitedRatio();
}
double
SimEnv::
GetDensity(np::ndarray np_array) {
	int dim = mRegressionMemory->GetDim();
	Eigen::VectorXd tp = DPhy::toEigenVector(np_array, dim);
	return mRegressionMemory->GetDensity(mRegressionMemory->Normalize(tp));
}
p::list 
SimEnv::
GetParamSpaceSummary() {
	std::tuple<std::vector<Eigen::VectorXd>, 
			   std::vector<double>, 
			   std::vector<double>> x_y_z = mRegressionMemory->GetParamSpaceSummary();

	np::ndarray x = DPhy::toNumPyArray(std::get<0>(x_y_z));
	np::ndarray y = DPhy::toNumPyArray(std::get<1>(x_y_z));
	np::ndarray z = DPhy::toNumPyArray(std::get<2>(x_y_z));

	p::list l;
	l.append(x);
	l.append(y);
	l.append(z);

	return l;
}

using namespace boost::python;

BOOST_PYTHON_MODULE(simEnv)
{
	Py_Initialize();
	np::initialize();

	class_<SimEnv>("Env",init<int, std::string, std::string, bool, bool>())
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
		.def("GetRewardsByParts",&SimEnv::GetRewardsByParts)
		.def("GetParamGoal",&SimEnv::GetParamGoal)
		.def("UniformSample",&SimEnv::UniformSample)
		.def("LoadAdaptiveMotion",&SimEnv::LoadAdaptiveMotion)
		.def("TrainRegressionNetwork",&SimEnv::TrainRegressionNetwork)
		.def("GetPhaseLength",&SimEnv::GetPhaseLength)
		.def("GetDOF",&SimEnv::GetDOF)
		.def("SetGoalParameters",&SimEnv::SetGoalParameters)
		.def("SaveParamSpace",&SimEnv::SaveParamSpace)
		.def("SaveParamSpaceLog",&SimEnv::SaveParamSpaceLog)
		.def("UpdateReference",&SimEnv::UpdateReference)
		.def("GetVisitedRatio",&SimEnv::GetVisitedRatio)
		.def("GetDensity",&SimEnv::GetDensity)
		.def("GetParamSpaceSummary",&SimEnv::GetParamSpaceSummary)
		.def("UpdateParamState",&SimEnv::UpdateParamState);

}