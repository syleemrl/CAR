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

	if(adaptive) {
		Py_Initialize();
		np::initialize();
		try{
			p::object regression = p::import("regression");
			this->mRegression = regression.attr("Regression")();
			this->mRegression.attr("initTrain")(training_path, 2, mReferenceManager->GetDOF());
		}
		catch (const p::error_already_set&)
		{
			PyErr_Print();
		}

		//manual setting
		mTargetInterval = 0.04;
		mMaxbound = std::pair<double, double>(1.1, 1.5);
		nBins = (int)std::ceil((mMaxbound.second - mMaxbound.first) / mTargetInterval);
		for(int i = 0; i < nBins; i++) {
			mTargetBin.push_back(0);
		}
		mFlag_max = false;
	}
	isAdaptive = adaptive;
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
	return t;
}
void 
SimEnv::
TrainRegressionNetwork()
{
	std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> x_y = mReferenceManager->GetRegressionSamples();
	if(!mFlag_max) {
		for(int i = 0; i < x_y.first.size(); i += mReferenceManager->GetNumCPS()) {
			double idx = ((x_y.first)[i](1) - mMaxbound.first) / mTargetInterval;
			idx = std::floor(idx);
			if(idx >= 0 && idx < nBins) {
				mTargetBin[idx] += 1;
			}
		}
	}
	np::ndarray x = DPhy::toNumPyArray(x_y.first);
	np::ndarray y = DPhy::toNumPyArray(x_y.second);
	
	p::list l;
	l.append(x);
	l.append(y);

	this->mRegression.attr("saveRegressionData")(l);
	this->mRegression.attr("updateRegressionData")(l);
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
p::list
SimEnv::
GetHindsightTuples()
{

	int nCps = mReferenceManager->GetNumCPS();
	int dof = mReferenceManager->GetDOF();
	p::list input_li;
	p::list result_li;
	std::vector<std::vector<Eigen::VectorXd>> targetParameters;
	for (int id = 0; id < mNumSlaves; ++id)
	{
		targetParameters.push_back(mSlaves[id]->GetHindsightTarget());
		int nInput = targetParameters[id].size()*nCps;

		p::tuple shape = p::make_tuple(nInput, targetParameters[id][0].rows() + 1);
		np::dtype dtype = np::dtype::get_builtin<float>();
		np::ndarray input = np::empty(shape, dtype);

		float* data = reinterpret_cast<float*>(input.get_data());
		
		int idx = 0;
		for(int i = 0; i < targetParameters[id].size(); i++)
		{
			for(int j = 0; j < nCps; j++) {
				data[idx++] = (float)j;
				for(int k = 0; k < targetParameters[id][i].rows(); k++)
					data[idx++] = (float)targetParameters[id][i][k];
			}
		}
		input_li.append(input);
	}

	p::object output_li = this->mRegression.attr("runBatch")(input_li);
	std::vector<std::vector<std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>>>> ss;
	std::vector<std::vector<std::vector<Eigen::VectorXd>>> cps_;

	for (int id = 0; id < mNumSlaves; ++id)
	{
		int nInput = targetParameters[id].size()*nCps;

		np::ndarray na = np::from_object(output_li[id]);
		Eigen::VectorXd output = DPhy::toEigenVector(na, mNumSlaves*dof*nInput);
		std::vector<std::vector<Eigen::VectorXd>> cps;
		for(int i = 0; i < targetParameters[id].size(); i++)
		{
			std::vector<Eigen::VectorXd> cps_phase;
			for(int j = 0; j < nCps; j++) {
				cps_phase.push_back(output.block(i*nCps*dof + j*dof, 0, dof, 1));
			}
			cps.push_back(cps_phase);
		}
		cps_.push_back(cps);
	}
#pragma omp parallel for
	for (int id = 0; id < mNumSlaves; ++id)
	{
		std::vector<std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>>> sar = mSlaves[id]->GetHindsightSAR(cps_[id]);
		ss.push_back(sar);
	}
	for (int id = 0; id < mNumSlaves; ++id)
	{
		auto sar = ss[id];
		for(int l = 0; l < sar.size(); l++) {
			p::list sar_episodes;

			for(int i = 0; i < sar[l].size(); i++) {
				p::list sar_tuples;

				sar_tuples.append(DPhy::toNumPyArray(std::get<0>(sar[l][i])));
				sar_tuples.append(DPhy::toNumPyArray(std::get<1>(sar[l][i])));
				sar_tuples.append(DPhy::toNumPyArray(std::get<2>(sar[l][i])));
				sar_tuples.append(std::get<3>(sar[l][i]));
				
				sar_episodes.append(sar_tuples);
			}
			result_li.append(sar_episodes);
		}
	}

	return result_li;
}
void 
SimEnv::
SetRefUpdateMode(bool t) {
	mReferenceManager->SetRefUpdateMode(t);
	if(t) {
		// load cps
		mReferenceManager->LoadAdaptiveMotion("updated");
		Eigen::VectorXd tp(1);
		tp << 1.45;
		for(int id = 0; id < mNumSlaves; ++id) {
			mSlaves[id]->SetTargetParameters(tp);
		}
	} else {
		mReferenceManager->SaveAdaptiveMotion("updated");
	}
}
void 
SimEnv::
SetTargetParameters(np::ndarray np_array) {

	Eigen::VectorXd tp = DPhy::toEigenVector(np_array, 1);
	tp /= 100;
	std::cout << tp.transpose() << std::endl;
	int dof = mReferenceManager->GetDOF();

	std::vector<Eigen::VectorXd> cps;
	for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
		Eigen::VectorXd input(2);
		input << j, tp(0);
		p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
		np::ndarray na = np::from_object(a);
		cps.push_back(DPhy::toEigenVector(na, dof));
	}

	mReferenceManager->LoadAdaptiveMotion(cps);
	for(int id = 0; id < mNumSlaves; ++id) {
		mSlaves[id]->SetTargetParameters(tp);
	}
}
p::list  
SimEnv::
GetTargetBound() {
	p::list bound;
	if(mFlag_max) {
		bound.append((int)(mMaxbound.first*100));
		bound.append((int)(mMaxbound.second*100));
	} else {
		double min = -1, max = nBins;
		for(int i = 0; i < nBins; i++) {
			if(mTargetBin[i] >= 30) {
				if(min == -1)
					min = i;
			} else if(min != -1) {
				max = i;
				break;
			}
		}
		if(min == 0 && max == nBins)
			mFlag_max = true;

		if(min == -1) {
			bound.append(0);
			bound.append(0);
		} else {
			bound.append((int)((mMaxbound.first + mTargetInterval * min)*100));
			bound.append((int)((mMaxbound.first + mTargetInterval * max)*100));
		}
		std::cout << "regression bins: ";
		for(int i = 0; i < nBins; i++) {
			std::cout << mTargetBin[i] << " ";
		}
		std::cout << std::endl;
	}
	return bound;
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
		.def("GetHindsightTuples",&SimEnv::GetHindsightTuples)
		.def("TrainRegressionNetwork",&SimEnv::TrainRegressionNetwork)
		.def("Optimize",&SimEnv::Optimize)
		.def("GetDOF",&SimEnv::GetDOF)
		.def("LoadAdaptiveMotion",&SimEnv::LoadAdaptiveMotion)
		.def("SetTargetParameters",&SimEnv::SetTargetParameters)
		.def("SetRefUpdateMode",&SimEnv::SetRefUpdateMode)
		.def("GetTargetBound",&SimEnv::GetTargetBound)
		.def("GetRewardsByParts",&SimEnv::GetRewardsByParts);
}