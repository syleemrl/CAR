#include "Character.h"
#include "SkeletonBuilder.h"
#include <tinyxml.h>
#include "Functions.h"
using namespace dart::dynamics;
namespace DPhy
{
/**
*
* @brief Constructor
* @details Initialize Character.
* @param path Character file path
* @param dof Character degree of freedom
*/ 
Character::Character(const std::string& path)
{
	std::pair<SkeletonPtr, std::map<std::string, double>*> p = SkeletonBuilder::BuildFromFile(path);
	this->mSkeleton = p.first;
	this->mTorqueMap = p.second;

	mPath = path;
}
// Character::Character(const dart::dynamics::SkeletonPtr& skeleton)
// {
// 	this->mSkeleton = skeleton;
// }

const dart::dynamics::SkeletonPtr& Character::GetSkeleton()
{
	return this->mSkeleton;
}
void Character::SetSkeleton(dart::dynamics::SkeletonPtr skel)
{
	this->mSkeleton = skel;
}
void Character::SetPDParameters(double kp, double kv)
{
	int dof = this->mSkeleton->getNumDofs();
	this->SetPDParameters(Eigen::VectorXd::Constant(dof, kp), Eigen::VectorXd::Constant(dof, kv));
}

void Character::SetPDParameters(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv)
{
	this->mKp = kp;
	this->mKv = kv;
	this->mKp.block<6,1>(0,0).setZero();
	this->mKv.block<6,1>(0,0).setZero();

	this->mKp_default = this->mKp;
	this->mKv_default = this->mKv;
}

void Character::SetPDParameters(const Eigen::VectorXd& k)
{
	int dof = this->mSkeleton->getNumDofs();
	this->mKp.setZero();
	this->mKv.setZero();
	this->mKp.segment(6, dof-6) = k.segment(0, dof-6).array()*this->mKp_default.segment(6, dof-6).array();
	this->mKv.segment(6, dof-6) = k.segment(dof-6, dof-6).array()*this->mKv_default.segment(6, dof-6).array();
}

double Character::GetTorqueLimit(const std::string name)
{
	return mTorqueMap->find(name)->second;
}

void Character::ApplyForces(const Eigen::VectorXd& forces)
{
	this->mSkeleton->setForces(forces);
}

Eigen::VectorXd Character::GetPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired)
{
	auto& skel = mSkeleton;
	int dof = this->mSkeleton->getNumDofs();
	Eigen::VectorXd dq = skel->getVelocities();
	double dt = skel->getTimeStep();
	Eigen::VectorXd p_diff(dof);
	p_diff = this->mSkeleton->getPositionDifferences(dq*dt, -1.0*this->mSkeleton->getPositions());
	p_diff = this->mSkeleton->getPositionDifferences(p_desired, p_diff);

	p_diff.segment<6>(0) = Eigen::VectorXd::Zero(6);
	for(int i = 6; i < skel->getNumDofs(); i+=3){
		Eigen::Vector3d v = p_diff.segment<3>(i);
		double angle = v.norm();
		Eigen::Vector3d axis = v.normalized();

		angle = RadianClamp(angle);
		p_diff.segment<3>(i) = angle * axis;
	}

	Eigen::VectorXd v_diff(dof);
	v_diff = v_desired - this->mSkeleton->getVelocities();

	Eigen::VectorXd tau = mKp.cwiseProduct(p_diff) + mKv.cwiseProduct(v_diff);

	tau.segment<6>(0) = Eigen::VectorXd::Zero(6);
	return tau;
}
Eigen::VectorXd Character::GetSPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired)
{
	auto& skel = mSkeleton;
	Eigen::VectorXd q = skel->getPositions();
	Eigen::VectorXd dq = skel->getVelocities();
	double dt = skel->getTimeStep();
	Eigen::MatrixXd M_inv = (skel->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();

	// Eigen::VectorXd p_d = q + dq*dt - p_desired;
	Eigen::VectorXd p_d(q.rows());
	// clamping radians to [-pi, pi], only for ball joints
	// TODO : make it for all type joints
	p_d.segment<6>(0) = Eigen::VectorXd::Zero(6);
	for(int i = 6; i < skel->getNumDofs(); i+=3){
		Eigen::Quaterniond q_s = DARTPositionToQuaternion(q.segment<3>(i));
		Eigen::Quaterniond dq_s = DARTPositionToQuaternion(dt*(dq.segment<3>(i)));
		Eigen::Quaterniond q_d_s = DARTPositionToQuaternion(p_desired.segment<3>(i));

		Eigen::Quaterniond p_d_s = q_d_s.inverse()*q_s*dq_s;

		Eigen::Vector3d v = QuaternionToDARTPosition(p_d_s);
		double angle = v.norm();
		if(angle > 1e-8){
			Eigen::Vector3d axis = v.normalized();

			angle = RadianClamp(angle);	
			p_d.segment<3>(i) = angle * axis;
		}
		else
			p_d.segment<3>(i) = v;
	}
	Eigen::VectorXd p_diff = -mKp.cwiseProduct(p_d);
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq-v_desired);
	Eigen::VectorXd qddot = M_inv*(-skel->getCoriolisAndGravityForces()+
							p_diff+v_diff+skel->getConstraintForces());

	Eigen::VectorXd tau = p_diff + v_diff - dt*mKv.cwiseProduct(qddot);
	tau.segment<6>(0) = Eigen::VectorXd::Zero(6);
	return tau;
}
void Character::LoadBVHMap()
{
	TiXmlDocument doc;
	if(!doc.LoadFile(mPath)){
		std::cout << "Can't open file : " << mPath << std::endl;
		return;
	}

	TiXmlElement *skeldoc = doc.FirstChildElement("Skeleton");
	
	std::string skelname = skeldoc->Attribute("name");

	for(TiXmlElement *body = skeldoc->FirstChildElement("Joint"); body != nullptr; body = body->NextSiblingElement("Joint")){
		// name
		std::string name = body->Attribute("name");

		// bvh
		if(body->Attribute("bvh")!=nullptr)
			mBVHMap.insert(std::make_pair(name,body->Attribute("bvh")));
	}
}



// void
// Character::
// ReadFramesFromBVH(BVH* bvh)
// {
// 	mBVHFrames.clear();
// 	avgCOMVelocity << 0, 0, 0;
// 	for(const auto ss :mBVHMap){
// 		bvh->AddMapping(ss.first,ss.second);
// 	}
// 	for(double t = 0; t < bvh->GetMaxTime(); t+=bvh->GetTimeStep())
// 	{
// 		int dof = mSkeleton->getPositions().rows();
// 		Eigen::VectorXd p = Eigen::VectorXd::Zero(dof);
// 		Eigen::VectorXd p1 = Eigen::VectorXd::Zero(dof);
// 		//Set p
// 		bvh->SetMotion(t);
// 		for(auto ss :mBVHMap)
// 		{
// 			dart::dynamics::BodyNode* bn = mSkeleton->getBodyNode(ss.first);
// 			Eigen::Matrix3d R = bvh->Get(ss.first);
// 			dart::dynamics::Joint* jn = bn->getParentJoint();
// 			Eigen::Vector3d a = dart::dynamics::BallJoint::convertToPositions(R);
// 			a = QuaternionToDARTPosition(DARTPositionToQuaternion(a));
// 			// p.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
// 			if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr
// 				|| dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){
// 				p.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
// 			}
// 			else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
// 				p[jn->getIndexInSkeleton(0)] = a[0];
// 				if(p[jn->getIndexInSkeleton(0)]>M_PI)
// 					p[jn->getIndexInSkeleton(0)] -= 2*M_PI;
// 				else if(p[jn->getIndexInSkeleton(0)]<-M_PI)
// 					p[jn->getIndexInSkeleton(0)] += 2*M_PI;
// 			}
// 		}
// 		p.block<3,1>(3,0) = bvh->GetRootCOM(); 
// 		//Set p1
// 		double prev_time;
// 		if( t < 0.05 )
// 			prev_time = t+0.05;
// 		else
// 			prev_time = t-0.05;

// 		bvh->SetMotion(prev_time);
// 		for(auto ss :mBVHMap)
// 		{
// 			dart::dynamics::BodyNode* bn = mSkeleton->getBodyNode(ss.first);
// 			Eigen::Matrix3d R = bvh->Get(ss.first);
// 			dart::dynamics::Joint* jn = bn->getParentJoint();

// 			Eigen::Vector3d a = dart::dynamics::BallJoint::convertToPositions(R);
// 			a = QuaternionToDARTPosition(DARTPositionToQuaternion(a));
// 			// p1.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
// 			if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr
// 				|| dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){
// 				p1.block<3,1>(jn->getIndexInSkeleton(0),0) = a;
// 			}
// 			else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
// 				p1[jn->getIndexInSkeleton(0)] = a[0];
// 				if(p1[jn->getIndexInSkeleton(0)]>M_PI)
// 					p1[jn->getIndexInSkeleton(0)] -= 2*M_PI;
// 				else if(p1[jn->getIndexInSkeleton(0)]<-M_PI)
// 					p1[jn->getIndexInSkeleton(0)] += 2*M_PI;
// 			}
// 		}
// 		p1.block<3,1>(3,0) = bvh->GetRootCOM();

// 		Eigen::VectorXd v;
// 		if( t < 0.05 ){
// 			v = mSkeleton->getPositionDifferences(p1, p)*20;
// 		}
// 		else{
// 			v = mSkeleton->getPositionDifferences(p, p1)*20;
// 		}

// 		for(auto& jn : mSkeleton->getJoints()){
// 			if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
// 				double v_ = v[jn->getIndexInSkeleton(0)];
// 				if(v_ > M_PI){
// 					v_ -= 2*M_PI;
// 				}
// 				else if(v_ < -M_PI){
// 					v_ += 2*M_PI;
// 				}
// 				v[jn->getIndexInSkeleton(0)] = v_;
// 			}
// 		}
// 		mBVHFrames.push_back(new Frame(p, v));
// 	}

// //calculate contact infomation
// 	double heightLimit = 0.05;
// 	double velocityLimit = 6;

// 	for(int i = 0; i < mBVHFrames.size(); i++)
// 	{
// 		Eigen::VectorXd contact(mContactList.size());
// 		contact.setZero();

// 		mSkeleton->setPositions(mBVHFrames[i]->position);
// 		mSkeleton->setVelocities(mBVHFrames[i]->velocity);
// 		mSkeleton->computeForwardKinematics(true,true,false);
		
// 		for(int j = 0; j < mContactList.size(); j++) 
// 		{
// 			double height = mSkeleton->getBodyNode(mContactList[j])->getWorldTransform().translation()[1];
// 			double velocity = mSkeleton->getBodyNode("FootEndR")->getLinearVelocity().norm();
// 			if(height < heightLimit && velocity < velocityLimit) {
// 				contact(j) = 1;
// 			} 

// 		}
// 		mBVHFrames[i]->SetContact(contact);
// 		mBVHFrames[i]->SetCOMposition(mSkeleton->getRootBodyNode()->getCOM());
// 		if(i != 0) { 
// 			mBVHFrames[i]->SetCOMvelocity(mSkeleton->getRootBodyNode()->getCOMLinearVelocity());
// 			avgCOMVelocity += mBVHFrames[i]->COMvelocity;
// 		}
// 	}

// 	mBVHFrames[0]->SetCOMvelocity(mBVHFrames[1]->COMvelocity);
// 	avgCOMVelocity += mBVHFrames[0]->COMvelocity;

// 	totalFrames = mBVHFrames.size();
// 	avgCOMVelocity /= totalFrames;

// 	mBVHFrames_r.clear();
// }
// void
// Character::
// RescaleOriginalBVH(double w)
// {
// 	mSkeleton->setPositions(mBVHFrames[0]->position);
// 	mSkeleton->setVelocities(mBVHFrames[0]->velocity);
// 	mSkeleton->computeForwardKinematics(true,true,false);

// 	double minheight = 0.0;
// 	for(int i = 0; i < mContactList.size(); i++) 
// 	{
// 		double height = mSkeleton->getBodyNode(mContactList[i])->getWorldTransform().translation()[1];
// 		if(i == 0 || height < minheight) minheight = height;
// 	}

// 	for(int i = 0; i < mBVHFrames.size(); i++)
// 	{
// 		Eigen::VectorXd p = mBVHFrames[i]->position;
// 		p[4] -= minheight - 0.02;
// 		mBVHFrames[i]->SetPosition(p);
// 	}

// //calculate contact infomation
// 	double heightLimit = 0.05;
// 	double velocityLimit = 6;
// 	Eigen::VectorXd prev_p;
// 	Eigen::VectorXd prev_v;
// 	for(int i = 0; i < mBVHFrames.size(); i++)
// 	{
// 		if(i != 0) {
// 			Eigen::VectorXd cur_p = mBVHFrames[i]->position;
// 			Eigen::Vector3d d_p = cur_p.segment<3>(3) - prev_p.segment<3>(3);
// 			d_p *= w;
// 			prev_p = cur_p;
// 			cur_p.segment<3>(3) = mBVHFrames[i-1]->position.segment<3>(3) + d_p;
// 			mBVHFrames[i]->SetPosition(cur_p);

// 			Eigen::VectorXd cur_v = mBVHFrames[i]->velocity;
// 			cur_v.segment<3>(3) = w * cur_v.segment<3>(3);

// 			mBVHFrames[i]->SetVelocity(cur_v);

// 		} else {
// 			prev_p = mBVHFrames[i]->position;
// 		}
// 		Eigen::VectorXd contact(mContactList.size());
// 		contact.setZero();

// 		mSkeleton->setPositions(mBVHFrames[i]->position);
// 		mSkeleton->setVelocities(mBVHFrames[i]->velocity);
// 		mSkeleton->computeForwardKinematics(true,true,false);
		
// 		for(int j = 0; j < mContactList.size(); j++) 
// 		{
// 			double height = mSkeleton->getBodyNode(mContactList[j])->getWorldTransform().translation()[1];
// 			double velocity = mSkeleton->getBodyNode("FootEndR")->getLinearVelocity().norm();
// 			if(height < heightLimit && velocity < velocityLimit) {
// 				contact(j) = 1;
// 			} 
// 		}
// 		mBVHFrames[i]->SetContact(contact);
// 		mBVHFrames[i]->SetCOMposition(mSkeleton->getRootBodyNode()->getCOM());
// 		Eigen::Vector3d v = mSkeleton->getRootBodyNode()->getCOMLinearVelocity();
// 		mBVHFrames[i]->SetCOMvelocity(v);
// 	}
// }
// Frame*
// Character::
// GetTargetPositionsAndVelocitiesFromBVH(BVH* bvh, double t, bool isPhase)
// {
// 	int bi = 5;
// 	int k0 = (int) std::floor(t);
// 	int k1 = (int) std::ceil(t);	
// 		if(mBVHFrames_r.size() == 0) {
// 			for(auto f: mBVHFrames) {
// 				mBVHFrames_r.push_back(new Frame(f));
// 			}
// 		}
// 		// blending : 이전 클립의 end - 4번째 frame부터 새 클립의 begin + 4 frame까지를 blending
// 		if(k1 / (mBVHFrames_r.size() - bi) < 1 ) {
// 			if (k0 == k1)
// 				return new Frame(mBVHFrames_r[k0]);
// 			else
// 				return new Frame(DPhy::BlendPosition(mBVHFrames_r[k1]->position, mBVHFrames_r[k0]->position, (t-k0)), DPhy::BlendPosition(mBVHFrames_r[k1]->velocity, mBVHFrames_r[k0]->velocity, (t-k0)));		
// 		}
// 		else {
// 			Eigen::Vector6d root_next = mBVHFrames[0]->position.segment<6>(0);
// 			Eigen::Vector6d root_prev = mBVHFrames_r[mBVHFrames_r.size() - 1]->position.segment<6>(0);

// 			Eigen::AngleAxisd root_next_ori(root_next.segment<3>(0).norm(), root_next.segment<3>(0).normalized());
// 			Eigen::AngleAxisd root_prev_ori(root_prev.segment<3>(0).norm(), root_prev.segment<3>(0).normalized());

// 			Eigen::Matrix3d root_dori;
// 			root_dori = root_prev_ori.inverse() * root_next_ori;
// 			root_dori = DPhy::projectToXZ(root_dori);
// 			std::vector<Eigen::VectorXd> positions;

// 			int prev_size = mBVHFrames_r.size();

// 			for(int i = 0; i < mBVHFrames.size(); i++) {
// 				Eigen::VectorXd position_next = mBVHFrames[i]->position;

// 				Eigen::Vector3d dpos = mBVHFrames[i]->position.segment<3>(3) - mBVHFrames[0]->position.segment<3>(3);
// 				dpos =  root_dori * dpos + root_prev.segment<3>(3);
// 				dpos[1] = position_next[4];

// 				Eigen::AngleAxisd cur_ori(mBVHFrames[i]->position.segment<3>(0).norm(), mBVHFrames[i]->position.segment<3>(0).normalized());
// 				Eigen::Matrix3d dori;
// 	//			cur_ori = root_next_ori.inverse() * cur_ori;
// 				dori = root_dori * cur_ori;
// 				Eigen::Quaterniond dori_q(dori);

// 				position_next.segment<3>(3) = dpos;
// 				position_next.segment<3>(0) = DPhy::QuaternionToDARTPosition(dori_q);

// 				positions.push_back(position_next);
// 				mBVHFrames_r.push_back(new Frame(positions[i], mBVHFrames[i]->velocity));

// 				// Eigen::VectorXd temp_v = mSkeleton->getPositionDifferences(mBVHFrames_r[mBVHFrames_r.size()-1]->position, mBVHFrames_r[mBVHFrames_r.size()-2]->position)* 1.0 / bvh->GetTimeStep();
// 				// std::cout << mBVHFrames[i]->velocity.transpose() << std::endl;
// 				// std::cout << temp_v.transpose() << std::endl;
// 			}
// 			for(int i = 0; i < bi; i++) {
// 				int idx = prev_size - (i + 1);
// 				double weight = 1.0 - (i+1) / (double)(bi+1);

// 				mBVHFrames_r[idx]->position = DPhy::BlendPosition(positions[0], mBVHFrames_r[idx]->position, weight);

// 				Eigen::VectorXd position_prev = DPhy::BlendPosition(mBVHFrames_r[idx]->position, mBVHFrames_r[idx-1]->position, 0.95);
// 				mBVHFrames_r[idx]->velocity = mSkeleton->getPositionDifferences(mBVHFrames_r[idx]->position, position_prev)* 20;
// 					// position_next[4] = weight * position_next[4] + (1 - weight) * position_prev[4]; 

// 			}
// 			if (k0 == k1)
// 				return new Frame(mBVHFrames_r[k0]);
// 			else
// 				return new Frame(DPhy::BlendPosition(mBVHFrames_r[k1]->position, mBVHFrames_r[k0]->position, (t-k0)), 
// 					DPhy::BlendPosition(mBVHFrames_r[k1]->velocity, mBVHFrames_r[k0]->velocity, (t-k0)));	
// 		}	
// }
};