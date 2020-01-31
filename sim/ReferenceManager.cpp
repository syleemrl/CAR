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
	
	mMotions.insert(std::make_pair("t", std::vector<Motion*>()));
	mMotions.insert(std::make_pair("b", std::vector<Motion*>()));
	mMotions_phase.insert(std::make_pair("t", std::vector<Motion*>()));
	mMotions_phase.insert(std::make_pair("b", std::vector<Motion*>()));
	mTorques_phase.clear();
	mWorks_phase.clear();
}
std::pair<bool, bool> ReferenceManager::CalculateContactInfo(Eigen::VectorXd p, Eigen::VectorXd v)
{
	double heightLimit = 0.05;
	double velocityLimit = 6;
	bool l, r;
	auto& skel = mCharacter->GetSkeleton();

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

	return std::pair<bool, bool>(l, r);

}
void ReferenceManager::LoadMotionFromBVH(std::string filename)
{
	int vec_size = mMotions.find("b")->second.size();
	for(int i = 0; i < vec_size; i++) {
		delete mMotions.find("b")->second.back();
		mMotions.find("b")->second.pop_back();
	}

	vec_size = mMotions_phase.find("b")->second.size();
	for(int i = 0; i < vec_size; i++) {
		delete mMotions_phase.find("b")->second.back();
		mMotions_phase.find("b")->second.pop_back();
	}


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
	for(double t = 0; t < bvh->GetMaxTime(); t+=bvh->GetTimeStep())
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

		mMotions_phase.find("b")->second.push_back(new Motion(p, v));
	}

	mPhaseLength = mMotions_phase.find("b")->second.size();
	mTimeStep = bvh->GetTimeStep();

	delete bvh;

}
void ReferenceManager::LoadMotionFromTrainedData(std::string filename)
{
	int vec_size = mMotions.find("t")->second.size();
	for(int i = 0; i < vec_size; i++) {
		delete mMotions.find("t")->second.back();
		mMotions.find("t")->second.pop_back();
	}

	vec_size = mMotions_phase.find("t")->second.size();
	for(int i = 0; i < vec_size; i++) {
		delete mMotions_phase.find("t")->second.back();
		mMotions_phase.find("t")->second.pop_back();
	}

	mTorques_phase.clear();
	mWorks_phase.clear();

	std::string path = std::string(CAR_DIR) + filename;
	std::ifstream is(path);
	char buffer[256];
	is >> buffer;
	mTimeStep = atof(buffer);

	is >> buffer;
	mPhaseLength = atoi(buffer);

	int dof = mCharacter->GetSkeleton()->getPositions().rows();

	for(int i = 0; i < mPhaseLength; i++) {
		Eigen::VectorXd t(dof);
		for(int j = 0; j < dof; j++) 
		{
			is >> buffer;
			t[j] = atof(buffer);
		}
		mTorques_phase.push_back(t);
	}

	mAvgWork = 0;
	for(int i = 0; i < mPhaseLength; i++) {
		double w;

		is >> buffer;
		w = atof(buffer);
	
		mWorks_phase.push_back(w);
		mAvgWork += w;
	}
	mAvgWork /= mPhaseLength;

	for(int i = 0; i < mPhaseLength; i++) {
		Eigen::VectorXd p(dof);
		for(int j = 0; j < dof; j++) 
		{
			is >> buffer;
			p[j] = atof(buffer);
		}
		mMotions_phase.find("t")->second.push_back(new Motion(p, Eigen::VectorXd(dof)));
	}

	for(int i = 0; i < mPhaseLength; i++) {
		Eigen::VectorXd v(dof);
		for(int j = 0; j < dof; j++) 
		{
			is >> buffer;
			v[j] = atof(buffer);
		}
		(mMotions_phase.find("t")->second)[i]->SetVelocity(v);
	}

	is.close();
	std::cout << "load trained data from: " << path << std::endl;
	std::cout << "phase length: " << mPhaseLength << std::endl;
}
void ReferenceManager::RescaleMotion(double w, std::string mode)
{
	auto& skel = mCharacter->GetSkeleton();

	skel->setPositions((mMotions_phase.find(mode)->second)[0]->GetPosition());
	skel->setVelocities((mMotions_phase.find(mode)->second)[0]->GetVelocity());
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

	for(int i = 0; i < mPhaseLength; i++)
	{
		Eigen::VectorXd p = (mMotions_phase.find(mode)->second)[i]->GetPosition();
		p[4] -= minheight - 0.02;
		(mMotions_phase.find(mode)->second)[i]->SetPosition(p);
	}

//calculate contact infomation
	double heightLimit = 0.05;
	double velocityLimit = 6;
	Eigen::VectorXd prev_p;
	Eigen::VectorXd prev_v;
	for(int i = 0; i < mPhaseLength; i++)
	{
		if(i != 0) {
			Eigen::VectorXd cur_p = (mMotions_phase.find(mode)->second)[i]->GetPosition();
			Eigen::Vector3d d_p = cur_p.segment<3>(3) - prev_p.segment<3>(3);
			d_p *= w;
			prev_p = cur_p;
			cur_p.segment<3>(3) = (mMotions_phase.find(mode)->second)[i-1]->GetPosition().segment<3>(3) + d_p;
			(mMotions_phase.find(mode)->second)[i]->SetPosition(cur_p);

			Eigen::VectorXd cur_v = (mMotions_phase.find(mode)->second)[i]->GetVelocity();
			cur_v.segment<3>(3) = w * cur_v.segment<3>(3);

			(mMotions_phase.find(mode)->second)[i]->SetVelocity(cur_v);

		} else {
			prev_p = (mMotions_phase.find(mode)->second)[i]->GetPosition();
		}

	}
}
void ReferenceManager::EditMotion(double w, std::string mode)
{
	auto& skel = mCharacter->GetSkeleton();
	int dof = mCharacter->GetSkeleton()->getPositions().rows();

	skel->setPositions((mMotions_phase.find(mode)->second)[0]->GetPosition());
	skel->setVelocities((mMotions_phase.find(mode)->second)[0]->GetVelocity());
	skel->computeForwardKinematics(true,true,false);
 	std::vector<Motion*> result;
	for(int i = 0; i < mPhaseLength; i++)
	{
		if(i != 0) {
			Eigen::VectorXd p(dof);
			Eigen::VectorXd v = (mMotions_phase.find(mode)->second)[i]->GetVelocity() * 0.033 * w;
			p = DPhy::RotatePosition(result[i-1]->GetPosition(), v);
			p.segment<3>(3) = (mMotions_phase.find(mode)->second)[i]->GetPosition().segment<3>(3);
			result.push_back(new Motion(p, v));
		} else {
			result.push_back(new Motion((mMotions_phase.find(mode)->second)[i]));
		}
	}

	int vec_size = mMotions.find(mode)->second.size();
	for(int i = 0; i < vec_size; i++) {
		delete mMotions.find(mode)->second.back();
		mMotions.find(mode)->second.pop_back();
	}

	vec_size = mMotions_phase.find(mode)->second.size();
	for(int i = 0; i < vec_size; i++) {
		delete mMotions_phase.find(mode)->second.back();
		mMotions_phase.find(mode)->second.pop_back();
	}

	for(int i = 0; i < mPhaseLength; i++)
	{
		mMotions_phase.find(mode)->second.push_back(result[i]);
	}

}
Motion* ReferenceManager::GetMotion(double t, std::string mode)
{
	auto& skel = mCharacter->GetSkeleton();

	if(mMotions_phase.find(mode)->second.size() == 0) {
		return new Motion(skel->getPositions(), skel->getVelocities());
	}

	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	
	if((mMotions.find(mode)->second).size() == 0) {
		for(auto m: mMotions_phase.find(mode)->second) {
			(mMotions.find(mode)->second).push_back(new Motion(m));
		}
	}
	if(k1 / ((mMotions.find(mode)->second).size() - mBlendingInterval) < 1 ) {
		if (k0 == k1)
			return new Motion((mMotions.find(mode)->second)[k0]);
		else
			return new Motion(DPhy::BlendPosition((mMotions.find(mode)->second)[k1]->GetPosition(), (mMotions.find(mode)->second)[k0]->GetPosition(), (t-k0)), 
				DPhy::BlendPosition((mMotions.find(mode)->second)[k1]->GetVelocity(), (mMotions.find(mode)->second)[k0]->GetVelocity(), (t-k0)));		
	}
	else {
		int prev_size = (mMotions.find(mode)->second).size();

		Eigen::Vector6d root_next = (mMotions_phase.find(mode)->second)[0]->GetPosition().segment<6>(0);
		Eigen::Vector6d root_prev = (mMotions.find(mode)->second)[prev_size - 1]->GetPosition().segment<6>(0);

		Eigen::AngleAxisd root_next_ori(root_next.segment<3>(0).norm(), root_next.segment<3>(0).normalized());
		Eigen::AngleAxisd root_prev_ori(root_prev.segment<3>(0).norm(), root_prev.segment<3>(0).normalized());
		Eigen::Matrix3d root_dori;
		root_dori = root_next_ori.inverse() * root_prev_ori;
		root_dori = DPhy::projectToXZ(root_dori);
		std::vector<Eigen::VectorXd> positions;

		for(int i = 0; i < mPhaseLength; i++) {
			Eigen::VectorXd position_next = (mMotions_phase.find(mode)->second)[i]->GetPosition();

			Eigen::Vector3d dpos = (mMotions_phase.find(mode)->second)[i]->GetPosition().segment<3>(3) - (mMotions_phase.find(mode)->second)[0]->GetPosition().segment<3>(3);
			dpos =  root_dori * dpos + root_prev.segment<3>(3);
			dpos[1] = position_next[4];

			Eigen::AngleAxisd cur_ori((mMotions_phase.find(mode)->second)[i]->GetPosition().segment<3>(0).norm(), (mMotions_phase.find(mode)->second)[i]->GetPosition().segment<3>(0).normalized());
			Eigen::Matrix3d dori;
	//			cur_ori = root_next_ori.inverse() * cur_ori;
			dori = root_dori * cur_ori;
			Eigen::Quaterniond dori_q(dori);

			position_next.segment<3>(3) = dpos;
			position_next.segment<3>(0) = DPhy::QuaternionToDARTPosition(dori_q);

			positions.push_back(position_next);
			(mMotions.find(mode)->second).push_back(new Motion(positions[i], (mMotions_phase.find(mode)->second)[i]->GetVelocity()));

				// Eigen::VectorXd temp_v = mSkeleton->getPositionDifferences(mBVHFrames_r[mBVHFrames_r.size()-1]->GetPosition(), mBVHFrames_r[mBVHFrames_r.size()-2]->GetPosition())* 1.0 / bvh->GetTimeStep();
				// std::cout << mBVHFrames[i]->GetVelocity().transpose() << std::endl;
				// std::cout << temp_v.transpose() << std::endl;
		}
		for(int i = mBlendingInterval-1; i >= 0; i--) {

			int idx = prev_size - (i + 1);
			double weight = 1.0 - (i+1) / (double)(mBlendingInterval+1);

			Eigen::VectorXd oldPosition =  (mMotions.find(mode)->second)[idx]->GetPosition();
			(mMotions.find(mode)->second)[idx]->SetPosition(DPhy::BlendPosition(positions[0], oldPosition, weight));

			Eigen::VectorXd position_prev = DPhy::BlendPosition((mMotions.find(mode)->second)[idx]->GetPosition(), (mMotions.find(mode)->second)[idx-1]->GetPosition(), 0.95);
			(mMotions.find(mode)->second)[idx]->SetVelocity(skel->getPositionDifferences((mMotions.find(mode)->second)[idx]->GetPosition(), position_prev)* 20);
			// position_next[4] = weight * position_next[4] + (1 - weight) * position_prev[4]; 

		}
		if (k0 == k1)
			return new Motion((mMotions.find(mode)->second)[k0]);
		else
			return new Motion(DPhy::BlendPosition((mMotions.find(mode)->second)[k1]->GetPosition(), (mMotions.find(mode)->second)[k0]->GetPosition(), (t-k0)), 
				DPhy::BlendPosition((mMotions.find(mode)->second)[k1]->GetVelocity(), (mMotions.find(mode)->second)[k0]->GetVelocity(), (t-k0)));	
	}	
}
};