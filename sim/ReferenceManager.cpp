#include "ReferenceManager.h"
#include <tinyxml.h>
#include <fstream>
#include <stdlib.h>

using namespace dart::dynamics;
namespace DPhy
{
ReferenceManager::ReferenceManager(Character* character)
{
	mCharacter = character;
	mBlendingInterval = 5;
	
	mMotions_gen.clear();
	mMotions_raw.clear();
	mMotions_phase.clear();
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
		//Set p1
		double prev_time;
		if( t < 0.05 )
			prev_time = t+0.05;
		else
			prev_time = t-0.05;

		bvh->SetMotion(prev_time);
		for(auto ss :bvhMap)
		{
			dart::dynamics::BodyNode* bn = skel->getBodyNode(ss.first);
			Eigen::Matrix3d R = bvh->Get(ss.first);
			dart::dynamics::Joint* jn = bn->getParentJoint();

			Eigen::Vector3d a = dart::dynamics::BallJoint::convertToPositions(R);
			a = QuaternionToDARTPosition(DARTPositionToQuaternion(a));
			// p1.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
			if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr
				|| dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){
				p1.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
			}
			else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
				p1[jn->getIndexInSkeleton(0)] = a[0];
				if(p1[jn->getIndexInSkeleton(0)]>M_PI)
					p1[jn->getIndexInSkeleton(0)] -= 2*M_PI;
				else if(p1[jn->getIndexInSkeleton(0)]<-M_PI)
					p1[jn->getIndexInSkeleton(0)] += 2*M_PI;
			}
		}
		p1.block<3,1>(3,0) = bvh->GetRootCOM();

		Eigen::VectorXd v;
		if( t < 0.05 ){
			v = skel->getPositionDifferences(p1, p)*20;
		}
		else{
			v = skel->getPositionDifferences(p, p1)*20;
		}
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
		mMotions_raw.push_back(new Motion(p, v));
		t += bvh->GetTimeStep();

	}

	mPhaseLength = mMotions_raw.size();
	mTimeStep = bvh->GetTimeStep();

	for(int i = 0; i < mPhaseLength; i++) {
		mMotions_phase.push_back(new Motion(mMotions_raw[i]));
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
void ReferenceManager::GenerateMotionsFromSinglePhase(int frames, bool blend)
{
	mMotions_gen.clear();

	auto& skel = mCharacter->GetSkeleton();

	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform(mMotions_phase[0]->GetPosition().head<6>());
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform(mMotions_phase.back()->GetPosition().head<6>());

	Eigen::Isometry3d T0_gen = T0_phase;
	
	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();

	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());			
	T01.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));
	T01.translation()[1] = 0;

	for(int i = 0; i < frames; i++) {
		
		int phase = i % mPhaseLength;
		
		if(i < mPhaseLength) {
			mMotions_gen.push_back(new Motion(mMotions_phase[i]));
		} else {
			Eigen::VectorXd pos = mMotions_phase[phase]->GetPosition();
			Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
			T_current = T0_phase.inverse()*T_current;
			T_current = T0_gen*T_current;

			pos.head<6>() = dart::dynamics::FreeJoint::convertToPositions(T_current);
			Eigen::VectorXd vel = skel->getPositionDifferences(pos, mMotions_gen.back()->GetPosition()) / 0.033;
			mMotions_gen.push_back(new Motion(pos, vel));

			if(blend && phase == 0) {
				for(int j = mBlendingInterval; j > 0; j--) {
					double weight = 1.0 - j / (double)(mBlendingInterval+1);
					Eigen::VectorXd oldPos = mMotions_gen[i - j]->GetPosition();
					mMotions_gen[i - j]->SetPosition(DPhy::BlendPosition(oldPos, pos, weight));
					vel = skel->getPositionDifferences(mMotions_gen[i - j]->GetPosition(), mMotions_gen[i - j - 1]->GetPosition()) / 0.033;
			 		mMotions_gen[i - j]->SetVelocity(vel);
				}
			}
		}
		if(phase == mPhaseLength - 1) {
			T0_gen = T01*T0_gen;

		}
	}

	// else {
	// 	int prev_size = (mMotions.find(mode)->second).size();

	// 	Eigen::Vector6d root_next = (mMotions_phase.find(mode)->second)[0]->GetPosition().segment<6>(0);
	// 	Eigen::Vector6d root_prev = (mMotions.find(mode)->second)[prev_size - 1]->GetPosition().segment<6>(0);

	// 	Eigen::AngleAxisd root_next_ori(root_next.segment<3>(0).norm(), root_next.segment<3>(0).normalized());
	// 	Eigen::AngleAxisd root_prev_ori(root_prev.segment<3>(0).norm(), root_prev.segment<3>(0).normalized());
	// 	Eigen::Matrix3d root_dori;
	// 	Eigen::Matrix3d root_ori_prev;
	// 	root_ori_prev = root_prev_ori;
	// 	root_dori = root_next_ori.inverse() * root_prev_ori;
	// 	root_dori = DPhy::projectToXZ(root_dori);
	// 	std::vector<Eigen::VectorXd> positions;

	// 	for(int i = 0; i < mPhaseLength; i++) {
	// 		Eigen::VectorXd position_next = (mMotions_phase.find(mode)->second)[i]->GetPosition();

	// 		Eigen::Vector3d dpos = (mMotions_phase.find(mode)->second)[i]->GetPosition().segment<3>(3) - (mMotions_phase.find(mode)->second)[0]->GetPosition().segment<3>(3);
	// 		dpos =  root_dori * dpos + root_prev.segment<3>(3);
	// 		dpos[1] = position_next[4];

	// 		Eigen::AngleAxisd cur_ori((mMotions_phase.find(mode)->second)[i]->GetPosition().segment<3>(0).norm(), (mMotions_phase.find(mode)->second)[i]->GetPosition().segment<3>(0).normalized());
	// 		Eigen::Matrix3d dori;
	// //			cur_ori = root_next_ori.inverse() * cur_ori;
	// 		dori = cur_ori * root_dori;
	// 		Eigen::Quaterniond dori_q(dori);

	// 		position_next.segment<3>(3) = dpos;
	// 		position_next.segment<3>(0) = DPhy::QuaternionToDARTPosition(dori_q);
	// 		positions.push_back(position_next);

	// 		Eigen::VectorXd velocity = skel->getPositionDifferences(position_next, (mMotions.find(mode)->second).back()->GetPosition())* 1.0 / 0.033;
	// 		(mMotions.find(mode)->second).push_back(new Motion(positions[i], velocity));

	// 			// Eigen::VectorXd temp_v = mSkeleton->getPositionDifferences(mBVHFrames_r[mBVHFrames_r.size()-1]->GetPosition(), mBVHFrames_r[mBVHFrames_r.size()-2]->GetPosition())* 1.0 / bvh->GetTimeStep();
	// 			// std::cout << mBVHFrames[i]->GetVelocity().transpose() << std::endl;
	// 			// std::cout << temp_v.transpose() << std::endl;
	// 	}
	// 	for(int i = mBlendingInterval-1; i >= 0; i--) {

	// 		int idx = prev_size - (i + 1);
	// 		double weight = 1.0 - (i+1) / (double)(mBlendingInterval+1);

	// 		Eigen::VectorXd oldPosition =  (mMotions.find(mode)->second)[idx]->GetPosition();
	// 		(mMotions.find(mode)->second)[idx]->SetPosition(DPhy::BlendPosition(positions[0], oldPosition, weight));
	// 		Eigen::VectorXd velocity = skel->getPositionDifferences((mMotions.find(mode)->second)[idx]->GetPosition(), (mMotions.find(mode)->second)[idx-1]->GetPosition()) * 1.0 / 0.033;
	// 		(mMotions.find(mode)->second)[idx]->SetVelocity(velocity);
	// 	}
	// 	if (k0 == k1)
	// 		return new Motion((mMotions.find(mode)->second)[k0]);
	// 	else
	// 		return new Motion(DPhy::BlendPosition((mMotions.find(mode)->second)[k1]->GetPosition(), (mMotions.find(mode)->second)[k0]->GetPosition(), (t-k0)), 
	// 			DPhy::BlendPosition((mMotions.find(mode)->second)[k1]->GetVelocity(), (mMotions.find(mode)->second)[k0]->GetVelocity(), (t-k0)));	
	// }	
}
Motion* ReferenceManager::GetMotion(double t)
{
	auto& skel = mCharacter->GetSkeleton();

	if(mMotions_gen.size()-1 < t) {
	 	return new Motion(mMotions_gen.back()->GetPosition(), mMotions_gen.back()->GetVelocity());
	}
	
	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	

	if (k0 == k1)
		return new Motion(mMotions_gen[k0]);
	else
		return new Motion(DPhy::BlendPosition(mMotions_gen[k1]->GetPosition(), mMotions_gen[k0]->GetPosition(), 1 - (t-k0)), 
				DPhy::BlendPosition(mMotions_gen[k1]->GetVelocity(), mMotions_gen[k0]->GetVelocity(), 1 - (t-k0)));		
}
};