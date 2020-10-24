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
		if(parametric) {
			mRegressionMemory = new DPhy::RegressionMemory();
			mReferenceManager->SetRegressionMemory(mRegressionMemory);
		}
		mReferenceManager->InitOptimization(num_slaves, training_path, parametric);
		mReferenceManager->LoadAdaptiveMotion();

		if(parametric) {
			mRegressionMemory->LoadParamSpace(mPath + "param_space");
			mReferenceManager->SetParamGoal(mRegressionMemory->GetParamGoal());
		}

	} else {
		mReferenceManager->InitOptimization(num_slaves, "");
	}

	if(parametric) {
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
	}
	
	mNumState = mSlaves[0]->GetNumState();
	mNumAction = mSlaves[0]->GetNumAction();
	mExUpdate = 0;
	isAdaptive = adaptive;
	isParametric = parametric;
	mNeedRefUpdate = true;
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
bool
SimEnv::
Optimize()
{
	bool t = mReferenceManager->Optimize();
	// int flag_sig = mReferenceManager->NeedUpdateSigTarget();
	// if(flag_sig) {
	// 	double sig = mSlaves[0]->GetSigTarget();
	// 	double new_sig = sig;
	// 	if(flag_sig > 0)
	// 		new_sig *= 2;
	// 	else if(flag_sig < 0 && sig > 0.3)
	// 		new_sig /= 2;
	// 	if(sig != new_sig) {
	// 		mReferenceManager->UpdateTargetReward(sig, new_sig);
	// 		for(int id = 0; id < mNumSlaves; ++id) {
	// 			mSlaves[id]->SetSigTarget(new_sig);
	// 		}		
	// 	}

	return t;
}
void 
SimEnv::
TrainRegressionNetwork(int n)
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
	this->mRegression.attr("train")(n);

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
SetExplorationMode(bool t) {

	mReferenceManager->SetExplorationMode(t);
	if(t) {
		// load cps
		mReferenceManager->LoadAdaptiveMotion("updated");
		Eigen::VectorXd tp = mReferenceManager->GetParamGoal();		
		for(int id = 0; id < mNumSlaves; ++id) {
			mSlaves[id]->SetGoalParameters(tp);
			// mSlaves[id]->SetExplorationMode(true);
		}
		mRegressionMemory->ResetPrevSpace();
	} else {
		mReferenceManager->SaveAdaptiveMotion("updated");
		for(int id = 0; id < mNumSlaves; ++id) {
			mSlaves[id]->SetExplorationMode(false);
		}
	}
}
int 
SimEnv::
NeedUpdateGoal() {
	if(mRegressionMemory->IsSpaceFullyExplored())
		return -1;

	if(mReferenceManager->CheckExplorationProgress())
		return 0;

	// Eigen::VectorXd goal_new = mRegressionMemory->SelectNewParamGoal();
	// mReferenceManager->SetParamGoal(goal_new);
	// for(int id = 0; id < mNumSlaves; ++id) {
	// 	mSlaves[id]->SetGoalParameters(goal_new);
	// }
	return 1;
}
bool 
SimEnv::
NeedParamTraining() {
	return mRegressionMemory->IsSpaceExpanded();
}
void 
SimEnv::
SetGoalParameters(np::ndarray np_array) {

	int dim = mRegressionMemory->GetDim();
	Eigen::VectorXd tp = DPhy::toEigenVector(np_array, dim);
	Eigen::VectorXd tp_normalized = mRegressionMemory->Normalize(tp);
	int dof = mReferenceManager->GetDOF() + 1;
	int dof_input = 1 + mRegressionMemory->GetDim();
	std::vector<Eigen::VectorXd> cps;
	for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
		Eigen::VectorXd input(dof_input);
		input << j, tp_normalized;
		p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
		np::ndarray na = np::from_object(a);
		cps.push_back(DPhy::toEigenVector(na, dof));
	}

	mReferenceManager->LoadAdaptiveMotion(cps);

	for(int id = 0; id < mNumSlaves; ++id) {
		mSlaves[id]->SetGoalParameters(tp);
	}
}
void
SimEnv::
SetExGoalParameters(np::ndarray np_array) {
	int dim = mRegressionMemory->GetDim();
	Eigen::VectorXd tp = DPhy::toEigenVector(np_array, dim);
	std::vector<Eigen::VectorXd> cps = mRegressionMemory->GetCPSFromNearestParams(tp);

	mReferenceManager->LoadAdaptiveMotion(cps);
	mReferenceManager->SaveAdaptiveMotion("ex_"+std::to_string(mExUpdate));

	for(int id = 0; id < mNumSlaves; ++id) {
		mSlaves[id]->SetGoalParameters(tp);
	}
	mReferenceManager->SetParamGoal(tp);
	mReferenceManager->ResetOptimizationParameters(false);
	mExUpdate += 1;
	std::ofstream ofs;
	ofs.open(mPath + "goal_ex", std::fstream::out | std::fstream::app);

	ofs << mExUpdate << " " << tp.transpose() << std::endl;
	ofs.close();
}
np::ndarray 
SimEnv::
GetParamGoal() {
	return DPhy::toNumPyArray(mReferenceManager->GetParamGoal());
}
np::ndarray 
SimEnv::
UniformSampleParam() {
	return DPhy::toNumPyArray(mRegressionMemory->UniformSample());
}
void
SimEnv::
SaveParamSpace() {
	mRegressionMemory->SaveParamSpace(mPath + "param_space");

}
void
SimEnv::
SaveParamSpaceLog(int n) {
	mRegressionMemory->SaveLog(mPath + "log");

}
p::list
SimEnv::
GetParamGoalCandidate() {
	std::vector<std::pair<Eigen::VectorXd, std::vector<Eigen::VectorXd>>> ps = mRegressionMemory->SelectNewParamGoalCandidate();
	p::list li;
	for(int i = 0; i < ps.size(); i++) {
		p::list p;
		p.append(DPhy::toNumPyArray(ps[i].first));
		p::list p_near;
		for(int j = 0 ; j < ps[i].second.size(); j++) {
			p_near.append(DPhy::toNumPyArray((ps[i].second)[j]));
		}
		p.append(p_near);
		li.append(p);
	}
	return li;
}
using namespace boost::python;

BOOST_PYTHON_MODULE(simEnv)
{
	Py_Initialize();
	np::initialize();

	class_<SimEnv>("Env",init<int, std::string, std::string, bool, bool>())
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
		.def("TrainRegressionNetwork",&SimEnv::TrainRegressionNetwork)
		.def("Optimize",&SimEnv::Optimize)
		.def("GetDOF",&SimEnv::GetDOF)
		.def("LoadAdaptiveMotion",&SimEnv::LoadAdaptiveMotion)
		.def("SetGoalParameters",&SimEnv::SetGoalParameters)
		.def("SetExplorationMode",&SimEnv::SetExplorationMode)
		.def("NeedUpdateGoal",&SimEnv::NeedUpdateGoal)
		.def("NeedParamTraining",&SimEnv::NeedParamTraining)
		.def("GetParamGoal",&SimEnv::GetParamGoal)
		.def("UniformSampleParam",&SimEnv::UniformSampleParam)
		.def("SetExGoalParameters",&SimEnv::SetExGoalParameters)
		.def("SaveParamSpace",&SimEnv::SaveParamSpace)
		.def("SaveParamSpaceLog",&SimEnv::SaveParamSpaceLog)
		.def("GetParamGoalCandidate",&SimEnv::GetParamGoalCandidate)
		.def("GetRewardsByParts",&SimEnv::GetRewardsByParts);

}