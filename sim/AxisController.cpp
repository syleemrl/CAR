#include "AxisController.h"
#include <iostream>
#include <fstream>

namespace DPhy
{	
void
AxisController::
SetIdxs(std::vector<double> idxs)
{
	mIdxs = idxs;
}
void
AxisController::
Initialize(ReferenceManager* referenceManager) {
	rewards = 1.5;
	mAxis.clear();
	for(int i = 0; i < referenceManager->GetPhaseLength(); i++) {
		mAxis.push_back(Eigen::VectorXd::Zero(mIdxs.size() * 3));
		mDev.push_back(Eigen::VectorXd::Zero(mIdxs.size()));	
	}
	int ndof;
	for(int i = 1; i < referenceManager->GetPhaseLength() + 1; i++) {
		Eigen::VectorXd prev_target_position = referenceManager->GetPosition(i-1);
		Eigen::VectorXd cur_target_position = referenceManager->GetPosition(i);
		ndof = prev_target_position.rows();

		for(int j = 0; j < mIdxs.size(); j++) {
			Eigen::Vector3d target_diff_local;
			if(mIdxs[j] == 3) {
				target_diff_local = cur_target_position.segment<3>(mIdxs[j]) - prev_target_position.segment<3>(mIdxs[j]);
				Eigen::AngleAxisd root_prev = Eigen::AngleAxisd(prev_target_position.segment<3>(0).norm(), prev_target_position.segment<3>(0).normalized());
				target_diff_local = root_prev.inverse() * target_diff_local;
			} else {
				target_diff_local = JointPositionDifferences(cur_target_position.segment<3>(mIdxs[j]), prev_target_position.segment<3>(mIdxs[j]));
			}
			mAxis[i % referenceManager->GetPhaseLength()].segment<3>(3 * j) = target_diff_local;
		}
	}
	this->UpdateDev();

	mTuples.clear();
	for(int i = 0; i < referenceManager->GetPhaseLength(); i++) {
		std::vector<std::pair<double, Eigen::VectorXd>> tuple_times;
		tuple_times.clear();
		mTuples.push_back(tuple_times);
	}

	mTuples_temp.clear();
	mPrevPosition.clear();
	mTargetReward.clear();

	for(int i = 0; i < mslaves; i++) {
		std::vector<std::tuple<double, double, Eigen::VectorXd>> tuple_slaves;	
		tuple_slaves.clear();
		mTuples_temp.push_back(tuple_slaves);
		mPrevPosition.push_back(Eigen::VectorXd::Zero(ndof));
		mTargetReward.push_back(0);
	}
	mStartPosition = referenceManager->GetPosition(0);
}
void 
AxisController::
SetStartPosition(int slave) {
	mPrevPosition[slave] = mStartPosition;
}
void
AxisController::
SaveTuple(double time, double interval, Eigen::VectorXd position, int slave) {
	Eigen::VectorXd axis(mIdxs.size() * 3);
	for(int j = 0; j < mIdxs.size(); j++) {
		Eigen::Vector3d target_diff_local;
		if(mIdxs[j] == 3) {

			target_diff_local = position.segment<3>(mIdxs[j]) - mPrevPosition[slave].segment<3>(mIdxs[j]);
			Eigen::AngleAxisd root_prev = Eigen::AngleAxisd(mPrevPosition[slave].segment<3>(0).norm(), mPrevPosition[slave].segment<3>(0).normalized());
			target_diff_local = root_prev.inverse() * target_diff_local;
		} else {
			target_diff_local = JointPositionDifferences(position.segment<3>(mIdxs[j]), mPrevPosition[slave].segment<3>(mIdxs[j]));
		}
		axis.segment<3>(j * 3) = target_diff_local;
	}
	mTuples_temp[slave].push_back(std::tuple<double, double, Eigen::VectorXd>(time, 0, axis / interval));
	mPrevPosition[slave] = position;

}
void
AxisController::
SetTargetReward(double reward, int slave) {
	mTargetReward[slave] = reward;
}
void 
AxisController::
EndPhase(int slave) {
	for(int i = mTuples_temp[slave].size() - 1; i >= 0; i--) {
		double reward = std::get<1>(mTuples_temp[slave][i]);
		if(reward != 0) break;
		std::get<1>(mTuples_temp[slave][i]) = mTargetReward[slave];
	}
}
void 
AxisController::
EndEpisode(int slave) {

	for(int i = 0; i < mTuples_temp[slave].size(); i++) {
		double time_d = std::get<0>(mTuples_temp[slave][i]);
		int time = static_cast <int> (std::round(time_d)) % mTuples.size();
		double reward = std::get<1>(mTuples_temp[slave][i]);
		Eigen::VectorXd axis = std::get<2>(mTuples_temp[slave][i]);
		if(reward != 0) {
			mTuples[time].push_back(std::pair<double, Eigen::VectorXd>(reward, axis));
		}
	}
	mTuples_temp[slave].clear();
}
void 
AxisController::
UpdateAxis() {
	for(int i = 0; i < mTuples.size(); i++) {
		// Eigen::VectorXd mean(mIdxs.size() * 3);
		// Eigen::VectorXd square_mean(mIdxs.size() * 3);
		// mean.setZero();
		// square_mean.setZero();
		// int count = 0;
		// for(int j = 0; j < mTuples[i].size(); j++) {
		// 	if(mTuples[i][j].first > rewards) {
		// 		mean += mTuples[i][j].second;
		// 		square_mean += mTuples[i][j].second.cwiseProduct(mTuples[i][j].second);
		// 		count += 1;
		// 	}
		// }
		// if(count >= 100) {
		// 	mean /= count;
		// 	square_mean /= count;
		// 	Eigen::VectorXd std_ewise = square_mean - mean.cwiseProduct(mean);
		// 	Eigen::VectorXd rates(mIdxs.size());
		// 	for(int k = 0; k < std_ewise.rows(); k+= 3) {
		// 		double std_mean;
		// 		std_mean = (std_ewise[k] + std_ewise[k+1] + std_ewise[k+2]) / 3.0;
		// 		double rate = std::min(0.3 * (1.0 / (std_mean * 1e5)), 0.3);
		// 		mAxis[i].segment<3>(k) = (1 - rate) * mAxis[i].segment<3>(k) + rate * mean.segment<3>(k);
		// 		rates(k / 3) = rate;
		// 	}

		// 	if(i == 53) {
		// 		std::cout << mAxis[i].transpose() << std::endl;
		// 		std::cout << mean.transpose() << std::endl;
		// 		std::cout << std_ewise.transpose() << std::endl;
		// 		std::cout << rates.transpose() << std::endl;
		// 	}
			mTuples[i].clear();
		//}
	}
	//this->UpdateDev();
}
void
AxisController::
UpdateDev() {
	int phaseLength = mAxis.size();
	for(int i = 0; i < phaseLength; i++) {
		int t = i + phaseLength;
		std::vector<std::pair<Eigen::VectorXd, double>> data;
		data.clear();
		for(int j = t - 3; j <= t + 3; j++) {
			int t_ = j % phaseLength;
			Eigen::VectorXd diff(mIdxs.size() * 3);
			Eigen::VectorXd y(mIdxs.size());
			for(int k = 0; k < mIdxs.size(); k++) {
				diff.segment<3>(k * 3) = mAxis[i].segment<3>(k * 3) - mAxis[t_].segment<3>(k * 3);
				double x = diff.segment<3>(k * 3).dot(mAxis[i].segment<3>(k * 3).normalized());
 				y(k) = (diff.segment<3>(k * 3) - x * mAxis[i].segment<3>(k * 3).normalized()).norm() / std::max(mAxis[i].segment<3>(k * 3).norm(), 0.0075);
			}
 			data.push_back(std::pair<Eigen::VectorXd, double>(y, (1 - abs(t - j) * 0.3)));
		}
	 	Eigen::VectorXd dev(mIdxs.size());
		dev.setZero();

		for(int i = 0; i < data.size(); i++) {
			int n = (int)(data[i].second * 100.0);
			Eigen::VectorXd y = data[i].first;
			dev += n * y.cwiseProduct(y);
		}
		mDev[i] = dev.cwiseSqrt();
	}

}
Eigen::VectorXd 
AxisController::
GetMean(double time) {
	int t = static_cast <int> (std::round(time)) % mAxis.size();
	return mAxis[t];
}
Eigen::VectorXd 
AxisController::
GetDev(double time) {
	int t = static_cast <int> (std::round(time)) % mDev.size();

	return mDev[t];
}
void 
AxisController::
Save(std::string path) {
	std::cout << "save axis and deviation to:" << path << std::endl;

	std::ofstream ofs(path);

	for(auto t: mAxis) {
		ofs << t.transpose() << std::endl;
	}
	for(auto t: mDev) {
		ofs << t.transpose() << std::endl;
	}
	ofs.close();

}
void 
AxisController::
Load(std::string path) {

	std::ifstream is(path);
	if(is.fail())
		return;
	std::cout << "load axis and deviation from: " << path << std::endl;

	char buffer[256];
	for(int i = 0; i < mAxis.size(); i++) {
		for(int j = 0; j < mIdxs.size() * 3; j++) 
		{
			is >> buffer;
			mAxis[i](j) = atof(buffer);
		}
	}
	for(int i = 0; i < mAxis.size(); i++) {
		for(int j = 0; j < mIdxs.size(); j++) 
		{
			is >> buffer;
			mDev[i](j) = atof(buffer);
		}
	}
	is.close();
}
}