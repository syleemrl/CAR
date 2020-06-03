#include "ReferenceManager.h"
#include <tinyxml.h>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#define TARGET_FRAME 44
using namespace dart::dynamics;
namespace DPhy
{
ReferenceManager::ReferenceManager(Character* character)
{
	mCharacter = character;
	mBlendingInterval = 10;
	
	mMotions_gen.clear();
	mMotions_raw.clear();
	mMotions_phase.clear();

	mContact_name.clear();
	mContact_name.push_back("FootEndR");
	mContact_name.push_back("FootR");
	mContact_name.push_back("FootEndL");
	mContact_name.push_back("FootL");

}
void
ReferenceManager::
SaveTuple(double time, Eigen::VectorXd position, int slave) {
	Eigen::VectorXd axis(mIdxs.size() * 3);
	if(time == 0 || time > TARGET_FRAME + 1) {
		mPrevPosition[slave] = position;
		return;
	}
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

	mTuples_temp[slave].push_back(std::tuple<double, double, Eigen::VectorXd>(time, 0, axis));
	mTuples_position[slave].push_back(position);
	mPrevPosition[slave] = position;

}
void
ReferenceManager::
InitializeAdaptiveSettings(std::vector<double> idxs, double nslaves, std::string path, bool saveTrajectory)
{
	mPath = path;
	mSaveTrajectory = saveTrajectory;
	mSlaves = nslaves;
	mIdxs = idxs;
	int ndof = mMotions_phase[0]->GetPosition().rows();
	mMotions_phase_adaptive.clear();
	for(int i = 0; i < this->GetPhaseLength(); i++) {
		mMotions_phase_adaptive.push_back(new Motion(mMotions_phase[i]));
	}
	this->GenerateMotionsFromSinglePhase(1000, false, true);

	mTuples.clear();
	for(int i = 0; i < this->GetPhaseLength(); i++) {
		std::vector<Eigen::VectorXd> tuple_times;
		tuple_times.clear();
		mTuples.push_back(tuple_times);
	}

	mTuples_temp.clear();
	mTuples_position.clear();
	mPrevPosition.clear();
	mTargetReward.clear();

	for(int i = 0; i < mSlaves; i++) {
		std::vector<std::tuple<double, double, Eigen::VectorXd>> tuple_slaves;	
		tuple_slaves.clear();
		mTuples_temp.push_back(tuple_slaves);
		mPrevPosition.push_back(Eigen::VectorXd::Zero(ndof));
		mTargetReward.push_back(0);

		std::vector<Eigen::VectorXd> tuple_slaves_pos;
		tuple_slaves_pos.clear();	
		mTuples_position.push_back(tuple_slaves_pos);
	}
	this->GetNewAxisFromMotion(false);
	this->GetNewAxisFromMotion(true);
	this->ComputeDeviation();
}
void
ReferenceManager::
GetNewAxisFromMotion(bool adaptive){
	std::vector<Motion*>* p_phase;
	std::vector<Motion*>* p_gen;
	std::vector<Eigen::VectorXd>* p_axis;

	if(adaptive)
	{
		p_phase = &mMotions_phase_adaptive;
		p_gen = &mMotions_gen_adaptive;
		p_axis = &mAxis;
	}
	else {
		p_phase = &mMotions_phase;
		p_gen = &mMotions_gen;
		p_axis = &mAxis_BVH;
	}

	(*p_axis).clear();
	for(int i = 0; i < (*p_phase).size(); i++) {
		Eigen::VectorXd m_cur = (*p_gen)[i]->GetPosition();
		Eigen::VectorXd m_next = (*p_gen)[i+1]->GetPosition();

		Eigen::VectorXd axis(mIdxs.size()*3);
		for(int j = 0; j < mIdxs.size(); j++) {
			if(mIdxs[j] == 3) {

				Eigen::AngleAxisd root_ori = Eigen::AngleAxisd(m_cur.segment<3>(0).norm(), m_cur.segment<3>(0).normalized());
				Eigen::Vector3d v = m_next.segment<3>(mIdxs[j]) - m_cur.segment<3>(mIdxs[j]);
				axis.segment<3>(j*3) = root_ori.inverse() * v;
			} else {
				Eigen::AngleAxisd joint_ori_cur = Eigen::AngleAxisd(m_cur.segment<3>(mIdxs[j]).norm(), m_cur.segment<3>(mIdxs[j]).normalized());
				Eigen::AngleAxisd joint_ori_next = Eigen::AngleAxisd(m_next.segment<3>(mIdxs[j]).norm(), m_next.segment<3>(mIdxs[j]).normalized());

				Eigen::AngleAxisd joint_ori_delta;
				joint_ori_delta = joint_ori_cur.inverse() * joint_ori_next;
				axis.segment<3>(j*3) = joint_ori_delta.axis() * joint_ori_delta.angle();
			}
		}
		(*p_axis).push_back(axis);
	}
}
void
ReferenceManager::
ComputeDeviation() {
	mDev_BVH.clear();
	for(int i = 0; i < mPhaseLength; i++) {
		int t = i + mPhaseLength;
		std::vector<std::pair<Eigen::VectorXd, double>> data;
		data.clear();
		for(int j = t - 3; j <= t + 3; j++) {
			int t_ = j % mPhaseLength;
			Eigen::VectorXd diff(mIdxs.size() * 3);
			Eigen::VectorXd y(mIdxs.size());
			for(int k = 0; k < mIdxs.size(); k++) {
				diff.segment<3>(k * 3) = mAxis_BVH[i].segment<3>(k * 3) - mAxis_BVH[t_].segment<3>(k * 3);
				double x = diff.segment<3>(k * 3).dot(mAxis_BVH[i].segment<3>(k * 3).normalized());
 				y(k) = (diff.segment<3>(k * 3) - x * mAxis_BVH[i].segment<3>(k * 3).normalized()).norm() / std::max(mAxis_BVH[i].segment<3>(k * 3).norm(), 0.0075);
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
		mDev_BVH.push_back(dev.cwiseSqrt());
	}
}
void
ReferenceManager::
GetNewMotionFromAxis(){

	for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
		int t_next = (i+1) % mPhaseLength;
		Eigen::VectorXd m_cur = mMotions_phase[i]->GetPosition();
		Eigen::VectorXd m_next = mMotions_phase[t_next]->GetPosition();

		for(int j = 0; j < mIdxs.size(); j++) {
			if(mIdxs[j] == 3) {
				Eigen::AngleAxisd root_ori = Eigen::AngleAxisd(m_cur.segment<3>(0).norm(), m_cur.segment<3>(0).normalized());
				m_next.segment<3>(mIdxs[j]) = m_cur.segment<3>(mIdxs[j]) + root_ori * mAxis[i].segment<3>(j*3);
			} else {
				Eigen::AngleAxisd joint_ori_cur = Eigen::AngleAxisd(m_cur.segment<3>(mIdxs[j]).norm(), m_cur.segment<3>(mIdxs[j]).normalized());
				Eigen::AngleAxisd joint_ori_delta = Eigen::AngleAxisd(mAxis[i].segment<3>(j*3).norm(), mAxis[i].segment<3>(j*3).normalized());

				Eigen::AngleAxisd joint_ori_next;
				joint_ori_next = joint_ori_cur * joint_ori_delta;
				m_next.segment<3>(mIdxs[j]) = joint_ori_next.axis() * joint_ori_next.angle();
				for(int k = 0; k < 3; k++) {
					if(m_next(mIdxs[j + k]) > M_PI)
						m_next(mIdxs[j + k]) -= 2*M_PI;
					if(m_next(mIdxs[j + k]) < -M_PI)
						m_next(mIdxs[j + k]) += 2*M_PI;
				}

			}
		}

		if(t_next != 0)
			mMotions_phase_adaptive[t_next]->SetPosition(m_next);
	
		Eigen::VectorXd vel = mCharacter->GetSkeleton()->getPositionDifferences(m_next, m_cur) / 0.033;
		mMotions_phase_adaptive[i]->SetVelocity(vel);
	}
}
void 
ReferenceManager::
CleanupMotion() {
	for(int i = 0; i < mContact_name.size(); i++)
		this->glueFootAndSmooth(i);
	for(int i = mBlendingInterval; i >= 0; i--) {
		int end = mMotions_phase.size() - 1;
		double weight = 1 - i / (double)(mBlendingInterval+1);
		Eigen::VectorXd oldPos = mMotions_phase[end - i]->GetPosition();
		Eigen::VectorXd newPos = mMotions_phase_adaptive[end - i]->GetPosition();
		newPos = DPhy::BlendPosition(newPos, oldPos, weight, false);
		mMotions_phase_adaptive[end - i]->SetPosition(newPos);
		Eigen::VectorXd newVel = mCharacter->GetSkeleton()->getPositionDifferences(mMotions_phase_adaptive[end - i]->GetPosition(), mMotions_phase_adaptive[end - i - 1]->GetPosition()) / 0.033;
		mMotions_phase_adaptive[end - i]->SetVelocity(newVel);
	}
}
void
ReferenceManager::
glueFootAndSmooth(int contact_idx) {
	std::vector<int> flag;
 	std::vector<std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>> constraints;
 	flag.clear();
 	constraints.clear();
	for(int i = 0; i < mPhaseLength; i++) {
		int flag_cur = 0;

		auto& skel = this->mCharacter->GetSkeleton();
		skel->setPositions(mMotions_phase_adaptive[i]->GetPosition());
		skel->computeForwardKinematics(true,false,false);

 		std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> constraints_cur;
 		constraints_cur.clear();

		Eigen::Vector3d p = skel->getBodyNode(mContact_name[contact_idx])->getWorldTransform().translation();
		if(mContact_BVH[i][contact_idx] && p[1] >= 0.06) {
			flag_cur = 1;
			p[1] = mContact_BVH[i][contact_idx];
			constraints_cur.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>(mContact_name[contact_idx], p, Eigen::Vector3d::Zero()));
		}

		if(constraints_cur.size() != 0) {
			Eigen::Vector3d p = mMotions_phase[i]->GetPosition().segment<3>(3);
			constraints_cur.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>("Torso", p, Eigen::Vector3d::Zero()));

			for(int j = 0; j < mContact_name.size(); j++) {
				int pair = contact_idx / 2;
				if(j / 2 != pair) {
					Eigen::Vector3d p = skel->getBodyNode(mContact_name[j])->getWorldTransform().translation();
					constraints_cur.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>(mContact_name[j], p, Eigen::Vector3d::Zero()));
				}
			}
		}

		flag.push_back(flag_cur);
		constraints.push_back(constraints_cur);
	}

	int start_idx = 0;
	int end_idx = 0;
	for(int i = 0; i < mPhaseLength; i++) {
		if(flag[i]) {
			auto& skel = this->mCharacter->GetSkeleton();
			skel->setPositions(mMotions_phase_adaptive[i]->GetPosition());
			skel->computeForwardKinematics(true,false,false);

			DPhy::solveMCIK(skel, constraints[i]);
			mMotions_phase_adaptive[i]->SetPosition(skel->getPositions());
		}
		if(i != 0 && mContact_BVH[i-1][contact_idx] != 0 && mContact_BVH[i][contact_idx] == 0) {
			end_idx = i-1;

			int d = mBlendingInterval / 2;

			for(int j = 1; j <= d; j++) {
				int idx = start_idx - j;
				if(idx < 0 || mContact_BVH[idx][contact_idx] != 0)
					break;

				double weight = 0.5*std::cos(0.5 * M_PI * (j / (d + 1.0)));
				Eigen::VectorXd newPos = DPhy::BlendPosition(mMotions_phase_adaptive[idx]->GetPosition(), mMotions_phase_adaptive[idx+1]->GetPosition(), weight, true);
				mMotions_phase_adaptive[idx]->SetPosition(newPos);
			}
			for(int j = 1; j <= d; j++) {
				int idx = end_idx + j - 1;
				if(idx >= mPhaseLength || mContact_BVH[idx][contact_idx] != 0)
					break;

				double weight = 0.5*std::cos(0.5 * M_PI * (j / (d + 1.0)));
				Eigen::VectorXd newPos = DPhy::BlendPosition(mMotions_phase_adaptive[idx]->GetPosition(), mMotions_phase_adaptive[idx-1]->GetPosition(), weight, true);

				mMotions_phase_adaptive[idx]->SetPosition(newPos);
			}
		}
		else if(i != 0 && mContact_BVH[i-1][contact_idx] == 0 && mContact_BVH[i][contact_idx] != 0){
			start_idx = i;
		}
	}
	for(int i = 1; i < mPhaseLength; i++) {
		Eigen::VectorXd newVel = mCharacter->GetSkeleton()->getPositionDifferences(mMotions_phase_adaptive[i]->GetPosition(), mMotions_phase_adaptive[i - 1]->GetPosition()) / 0.033;
		mMotions_phase_adaptive[i]->SetVelocity(newVel);
	}
}
void 
ReferenceManager::
UpdateMotion() {
	std::vector<Eigen::VectorXd> delta;
	delta.clear();

	int adaptive_ndof = mIdxs.size() * 3;
	for(int i = 0; i <= TARGET_FRAME; i++) {
		Eigen::VectorXd mean(adaptive_ndof + 1);
		Eigen::VectorXd square_mean(adaptive_ndof + 1);
		mean.setZero();
		square_mean.setZero();

		for(int j = 0; j < mTuples[i].size(); j++) {
			mean += mTuples[i][j];
			square_mean += mTuples[i][j].cwiseProduct(mTuples[i][j]);
		}
		mean /= mTuples[i].size();
		square_mean /= mTuples[i].size();

		Eigen::VectorXd var = square_mean - mean.cwiseProduct(mean);
		Eigen::VectorXd std = var.cwiseSqrt();
		Eigen::VectorXd covar(adaptive_ndof);
		covar.setZero();

		int update_count = 0;
		Eigen::VectorXd update_mean(adaptive_ndof);
		update_mean.setZero();
		if(i >= 40) {
			std::cout << i << std::endl;
		}
		for(int j = 0; j < mTuples[i].size(); j++) {
			double mean_diff = abs(mean[mean.rows()-1] - 1.3);
			double data_diff = abs(mTuples[i][j][mean.rows()-1] - 1.3);
			covar += (mTuples[i][j].head(adaptive_ndof) - mean.head(adaptive_ndof)) 
				* (data_diff - mean_diff) * 1.0 / mTuples[i].size();
			if(i >= 40) {
				std::cout << mean_diff << " " << data_diff << " " <<mTuples[i][j].segment<3>(0).transpose() << std::endl;
			}
			if(mean_diff > data_diff*1.2 || data_diff <= 0.03) {
				update_count += 1;
				update_mean += mTuples[i][j].head(adaptive_ndof);
			}
		}
		update_mean /= update_count;
		Eigen::VectorXd pearson_coef = covar.array() / (std.head(adaptive_ndof) * std.tail(1)).array();
		if(update_count >= 100) {
			if(i >= 40) {
				std::cout << i << std::endl;
				std::cout << pearson_coef.segment<3>(0).transpose() << std::endl;
				std::cout << mAxis[i].segment<3>(0).transpose() << std::endl;
				std::cout << mean.segment<3>(0).transpose() << std::endl;
				std::cout << update_mean.segment<3>(0).transpose() << std::endl;
			}

			for(int j = 0; j < adaptive_ndof; j++) {
				if(abs(pearson_coef(j)) < 0.05)
					continue;
				double rate = 0.5 * abs(pearson_coef(j));
				(mAxis[i])(j) = rate * update_mean(j) + (1.0 - rate) * (mAxis[i])(j);
			}
			this->SaveTuples(mTuples[i], std::to_string(i), 50);	
			mTuples[i].clear();
		}
	}
	this->GetNewMotionFromAxis();
	this->CleanupMotion();
	//this->GetNewAxisFromMotion();

	this->GenerateMotionsFromSinglePhase(1000, false, true);
}
void
ReferenceManager::
SetTargetReward(double reward, int slave) {
	mTargetReward[slave] = reward;
}
void 
ReferenceManager::
EndPhase(int slave) {
	for(int i = mTuples_temp[slave].size() - 1; i >= 0; i--) {
		double reward = std::get<1>(mTuples_temp[slave][i]);
		if(reward != 0) break;
		std::get<1>(mTuples_temp[slave][i]) = mTargetReward[slave];
	}
}
void 
ReferenceManager::
EndEpisode(int slave) {
	for(int i = 0; i < mTuples_temp[slave].size(); i++) {
		double time_d = std::get<0>(mTuples_temp[slave][i]);
		int time = static_cast <int> (std::round(time_d)) % mTuples.size();
		double reward = std::get<1>(mTuples_temp[slave][i]);
		Eigen::VectorXd vec(mIdxs.size() * 3 + 1);
		vec.head(mIdxs.size() * 3) = std::get<2>(mTuples_temp[slave][i]);
		vec[mIdxs.size() * 3] = reward;
		if(reward != 0) {
			mLock.lock();
			mTuples[time].push_back(vec);
			mLock.unlock();
		}
	}
	if(mSaveTrajectory)
		this->SaveEliteTrajectories(slave);
	mTuples_position[slave].clear();
	mTuples_temp[slave].clear();
}
void 
ReferenceManager::
SaveTuples(std::vector<Eigen::VectorXd> vecs, std::string postfix, int n) {
	std::string path = mPath + postfix;
	std::cout << "save motion to:" << path << std::endl;

	std::ofstream ofs(path, std::ios_base::app);

	int count = 0;
	for(auto t: vecs) {
		ofs << t.transpose() << std::endl;
		count++;
		if(count == n) 
			break;
	}
	ofs.close();

}
void 
ReferenceManager::
SaveAdaptiveMotion() {
	std::string path = mPath + std::string("adaptive");
	std::cout << "save motion to:" << path << std::endl;

	std::ofstream ofs(path);

	for(int i = 0; i < mAxis.size(); i++) {
		// ofs << mMotions_phase_adaptive[i]->GetPosition().transpose() << std::endl;
		// ofs << mMotions_phase_adaptive[i]->GetVelocity().transpose() << std::endl;
		ofs << mAxis[i].transpose() << std::endl;
	}
	ofs.close();

}
void 
ReferenceManager::
LoadAdaptiveMotion() {
	std::string path = mPath + std::string("adaptive");
	std::ifstream is(path);
	if(is.fail())
		return;
	std::cout << "load Motion from: " << path << std::endl;

	char buffer[256];
	for(int i = 0; i < mPhaseLength; i++) {
		Eigen::VectorXd pos(mMotions_phase[0]->GetPosition().rows());
		Eigen::VectorXd vel(mMotions_phase[0]->GetPosition().rows());
		// for(int j = 0; j < mMotions_phase[0]->GetPosition().rows(); j++) 
		// {
		// 	is >> buffer;
		// 	pos[j] = atof(buffer);
		// }
		// for(int j = 0; j < mMotions_phase[0]->GetVelocity().rows(); j++) 
		// {
		// 	is >> buffer;
		// 	vel[j] = atof(buffer);
		// }
		for(int j = 0; j < mAxis[0].rows(); j++) 
		{
			is >> buffer;
			mAxis[i][j] = atof(buffer);
		}
	//	mMotions_phase_adaptive[i]->SetPosition(pos);
	//	mMotions_phase_adaptive[i]->SetVelocity(vel);
	}
	is.close();
	this->GetNewMotionFromAxis();
	this->CleanupMotion();

	this->GenerateMotionsFromSinglePhase(1000, false, true);
//	this->GetNewAxisFromMotion();
}
std::pair<bool, bool> ReferenceManager::CalculateContactInfo(Eigen::VectorXd p, Eigen::VectorXd v)
{
	double heightLimit = 0.05;
	double velocityLimit = 6;
	bool l, r;

	auto& skel = mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v_save = mCharacter->GetSkeleton()->getVelocities();

	skel->setPositions(p);
	skel->setVelocities(v);
	skel->computeForwardKinematics(true,true,false);

	double height_f = skel->getBodyNode("FootR")->getWorldTransform().translation()[1];
	double velocity_f = skel->getBodyNode("FootR")->getLinearVelocity().norm();
	double height_fe = skel->getBodyNode("FootEndR")->getWorldTransform().translation()[1];
	double velocity_fe = skel->getBodyNode("FootEndR")->getLinearVelocity().norm();
		
	if(height_fe < heightLimit && velocity_fe < velocityLimit) {
		r = true; 
	} else if (height_f < heightLimit && velocity_f < velocityLimit) {
		r = true; 
	} else {
		r = false;
	}
		
	height_f = skel->getBodyNode("FootL")->getWorldTransform().translation()[1];
	velocity_f = skel->getBodyNode("FootL")->getLinearVelocity().norm();
	height_fe = skel->getBodyNode("FootEndL")->getWorldTransform().translation()[1];
	velocity_fe = skel->getBodyNode("FootEndL")->getLinearVelocity().norm();

	if(height_fe < heightLimit && velocity_fe < velocityLimit) {
		l = true; 
	} else if (height_f < heightLimit && velocity_f < velocityLimit) {
		l = true; 
	} else {
		l = false;
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	return std::pair<bool, bool>(l, r);

}
void ReferenceManager::LoadMotionFromBVH(std::string filename)
{
	mMotions_raw.clear();
	mMotions_phase.clear();
	
	this->mCharacter->LoadBVHMap();

	BVH* bvh = new BVH();
	std::string path = std::string(CAR_DIR) + filename;
	bvh->Parse(path);
	std::cout << "load trained data from: " << path << std::endl;

	auto& skel = mCharacter->GetSkeleton();
	std::map<std::string,std::string> bvhMap = mCharacter->GetBVHMap(); 
	for(const auto ss :bvhMap){
		bvh->AddMapping(ss.first,ss.second);
	}
	double t = 0;
	for(int i = 0; i < bvh->GetMaxFrame(); i++)
	{
		int dof = skel->getPositions().rows();
		Eigen::VectorXd p = Eigen::VectorXd::Zero(dof);
		Eigen::VectorXd p1 = Eigen::VectorXd::Zero(dof);
		//Set p
		bvh->SetMotion(t);
		for(auto ss :bvhMap)
		{
			dart::dynamics::BodyNode* bn = skel->getBodyNode(ss.first);
			Eigen::Matrix3d R = bvh->Get(ss.first);
			dart::dynamics::Joint* jn = bn->getParentJoint();
			Eigen::Vector3d a = dart::dynamics::BallJoint::convertToPositions(R);
			a = QuaternionToDARTPosition(DARTPositionToQuaternion(a));
			// p.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
			if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr
				|| dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){
				p.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
			}
			else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
				p[jn->getIndexInSkeleton(0)] = a[0];
				if(p[jn->getIndexInSkeleton(0)]>M_PI)
					p[jn->getIndexInSkeleton(0)] -= 2*M_PI;
				else if(p[jn->getIndexInSkeleton(0)]<-M_PI)
					p[jn->getIndexInSkeleton(0)] += 2*M_PI;
			}
		}
		p.block<3,1>(3,0) = bvh->GetRootCOM(); 

		Eigen::VectorXd v;
		if(t != 0)
		{
			v = skel->getPositionDifferences(p, mMotions_raw.back()->GetPosition()) / 0.033;
			for(auto& jn : skel->getJoints()){
				if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
					double v_ = v[jn->getIndexInSkeleton(0)];
					if(v_ > M_PI){
						v_ -= 2*M_PI;
					}
					else if(v_ < -M_PI){
						v_ += 2*M_PI;
					}
					v[jn->getIndexInSkeleton(0)] = v_;
				}
			}
			mMotions_raw.back()->SetVelocity(v);
		}
		mMotions_raw.push_back(new Motion(p, Eigen::VectorXd(p.rows())));

		t += bvh->GetTimeStep();

	}

	mPhaseLength = mMotions_raw.size();
	mTimeStep = bvh->GetTimeStep();

	for(int i = 0; i < mPhaseLength; i++) {
		mMotions_phase.push_back(new Motion(mMotions_raw[i]));

		auto& skel = this->mCharacter->GetSkeleton();
		skel->setPositions(mMotions_phase.back()->GetPosition());
		skel->computeForwardKinematics(true,false,false);

		std::vector<double> result;
		result.clear();
		for(int i = 0; i < mContact_name.size(); i++) {
			Eigen::Vector3d p = skel->getBodyNode(mContact_name[i])->getWorldTransform().translation();
			if(p[1] < 0.06) {
				result.push_back(p[1]);
			} else {
				result.push_back(0);
			}
		}
		mContact_BVH.push_back(result);
	}

	delete bvh;

}
void ReferenceManager::RescaleMotion(double w)
{
	mMotions_phase.clear();

	auto& skel = mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v_save = mCharacter->GetSkeleton()->getVelocities();

	skel->setPositions(mMotions_raw[0]->GetPosition());
	skel->setVelocities(mMotions_raw[0]->GetVelocity());
	skel->computeForwardKinematics(true,true,false);

	double minheight = 0.0;
	std::vector<std::string> contactList;
	contactList.push_back("FootR");
	contactList.push_back("FootL");
	contactList.push_back("FootEndR");
	contactList.push_back("FootEndL");
	contactList.push_back("HandR");
	contactList.push_back("HandL");
	
	for(int i = 0; i < contactList.size(); i++) 
	{
		double height = skel->getBodyNode(contactList[i])->getWorldTransform().translation()[1];
		if(i == 0 || height < minheight) minheight = height;
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	for(int i = 0; i < mPhaseLength; i++)
	{
		Eigen::VectorXd p = mMotions_raw[i]->GetPosition();
		p[4] -= minheight - 0.02;
		mMotions_phase[i]->SetPosition(p);
	}

//calculate contact infomation
	double heightLimit = 0.05;
	double velocityLimit = 6;
	Eigen::VectorXd prev_p;
	Eigen::VectorXd prev_v;
	for(int i = 0; i < mPhaseLength; i++)
	{
		if(i != 0) {
			Eigen::VectorXd cur_p = mMotions_raw[i]->GetPosition();
			Eigen::Vector3d d_p = cur_p.segment<3>(3) - prev_p.segment<3>(3);
			d_p *= w;
			prev_p = cur_p;
			cur_p.segment<3>(3) = mMotions_raw[i-1]->GetPosition().segment<3>(3) + d_p;
			mMotions_phase[i]->SetPosition(cur_p);

			Eigen::VectorXd cur_v = mMotions_raw[i]->GetVelocity();
			cur_v.segment<3>(3) = w * cur_v.segment<3>(3);

			mMotions_phase[i]->SetVelocity(cur_v);

		} else {
			prev_p = mMotions_raw[i]->GetPosition();
			mMotions_phase[i]->SetPosition(mMotions_raw[i]->GetPosition());
			mMotions_phase[i]->SetVelocity(mMotions_raw[i]->GetVelocity());
		}

	}
}
void ReferenceManager::GenerateMotionsFromSinglePhase(int frames, bool blend, bool adaptive)
{
	std::vector<Motion*>* p_phase;
	std::vector<Motion*>* p_gen;
	if(adaptive)
	{
		p_phase = &mMotions_phase_adaptive;
		p_gen = &mMotions_gen_adaptive;
	}
	else {
		p_phase = &mMotions_phase;
		p_gen = &mMotions_gen;
	}

	(*p_gen).clear();

	auto& skel = mCharacter->GetSkeleton();

	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform((*p_phase)[0]->GetPosition().head<6>());
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform((*p_phase).back()->GetPosition().head<6>());

	Eigen::Isometry3d T0_gen = T0_phase;
	
	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();

	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());			
	T01.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));
	T01.translation()[1] = 0;

	for(int i = 0; i < frames; i++) {
		
		int phase = i % mPhaseLength;
		
		if(i < mPhaseLength) {
			(*p_gen).push_back(new Motion((*p_phase)[i]));
		} else {
			Eigen::VectorXd pos = (*p_phase)[phase]->GetPosition();
			Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
			T_current = T0_phase.inverse()*T_current;
			T_current = T0_gen*T_current;

			pos.head<6>() = dart::dynamics::FreeJoint::convertToPositions(T_current);
			Eigen::VectorXd vel = skel->getPositionDifferences(pos, (*p_gen).back()->GetPosition()) / 0.033;
			(*p_gen).back()->SetVelocity(vel);
			(*p_gen).push_back(new Motion(pos, vel));

			if(blend && phase == 0) {
				for(int j = mBlendingInterval; j > 0; j--) {
					double weight = 1.0 - j / (double)(mBlendingInterval+1);
					Eigen::VectorXd oldPos = (*p_gen)[i - j]->GetPosition();
					(*p_gen)[i - j]->SetPosition(DPhy::BlendPosition(oldPos, pos, weight));
					vel = skel->getPositionDifferences((*p_gen)[i - j]->GetPosition(), (*p_gen)[i - j - 1]->GetPosition()) / 0.033;
			 		(*p_gen)[i - j - 1]->SetVelocity(vel);
				}
			}
		}
		if(phase == mPhaseLength - 1) {
			T0_gen = T01*T0_gen;

		}
	}
}
Eigen::VectorXd ReferenceManager::GetPosition(double t , bool adaptive) 
{
	std::vector<Motion*>* p_gen;
	if(adaptive)
	{
		p_gen = &mMotions_gen_adaptive;
	}
	else {
		p_gen = &mMotions_gen;
	}

	auto& skel = mCharacter->GetSkeleton();

	if((*p_gen).size()-1 < t) {
	 	return (*p_gen).back()->GetPosition();
	}
	
	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	

	if (k0 == k1)
		return (*p_gen)[k0]->GetPosition();
	else
		return DPhy::BlendPosition((*p_gen)[k1]->GetPosition(), (*p_gen)[k0]->GetPosition(), 1 - (t-k0));	
}
Motion* ReferenceManager::GetMotion(double t, bool adaptive)
{
	std::vector<Motion*>* p_gen;
	if(adaptive)
	{
		p_gen = &mMotions_gen_adaptive;
	}
	else {
		p_gen = &mMotions_gen;
	}

	auto& skel = mCharacter->GetSkeleton();

	if(mMotions_gen.size()-1 < t) {
	 	return new Motion((*p_gen).back()->GetPosition(), (*p_gen).back()->GetVelocity());
	}
	
	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	

	if (k0 == k1)
		return new Motion((*p_gen)[k0]);
	else
		return new Motion(DPhy::BlendPosition((*p_gen)[k1]->GetPosition(), (*p_gen)[k0]->GetPosition(), 1 - (t-k0)), 
				DPhy::BlendPosition((*p_gen)[k1]->GetVelocity(), (*p_gen)[k0]->GetVelocity(), 1 - (t-k0)));		
}
void 
ReferenceManager::
SaveEliteTrajectories(int slave) {
	std::string path = mPath + std::string("trajectories");
	// std::cout << "save elite tuples to:" << path << std::endl;
	std::ofstream ofs(path, std::ios_base::app);
	double prev_reward = 0;

	for(int i = 0; i < mTuples_temp[slave].size(); i++) {
		if(std::get<1>(mTuples_temp[slave][i]) != prev_reward) {
			prev_reward = std::get<1>(mTuples_temp[slave][i]); 
		//	std::cout << prev_reward << std::endl;
		}
		if(std::get<1>(mTuples_temp[slave][i]) <= 0.5) {
			ofs << std::get<2>(mTuples_temp[slave][i]).transpose() << std::endl;
		}
	}
	ofs.close();
}
Eigen::VectorXd 
ReferenceManager::
GetAxis(double t) {
	int k0 = (int) std::floor(t);
	if(k0 == mPhaseLength)
		k0 -= 1;
	return mAxis_BVH[k0];
}
Eigen::VectorXd 
ReferenceManager::
GetDev(double t) {
	int k0 = (int) std::floor(t);
	if(k0 == mPhaseLength)
		k0 -= 1;
	return mDev_BVH[k0];
}
};