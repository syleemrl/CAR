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
	this->mSkeleton = DPhy::SkeletonBuilder::BuildFromFile(path);
	
	//temp
	mContactList.push_back("FootR");
	mContactList.push_back("FootL");
	mContactList.push_back("FootEndR");
	mContactList.push_back("FootEndL");
	mContactList.push_back("HandR");
	mContactList.push_back("HandL");

}
Character::Character(const dart::dynamics::SkeletonPtr& skeleton)
{
	this->mSkeleton = skeleton;

	//temp
	mContactList.push_back("FootR");
	mContactList.push_back("FootL");
	mContactList.push_back("FootEndR");
	mContactList.push_back("FootEndL");
	mContactList.push_back("HandR");
	mContactList.push_back("HandL");
}

const dart::dynamics::SkeletonPtr& Character::GetSkeleton()
{
	return this->mSkeleton;
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
void Character::LoadBVHMap(const std::string& path)
{
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
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
void
Character::
ReadFramesFromBVH(BVH* bvh)
{
	for(const auto ss :mBVHMap){
		bvh->AddMapping(ss.first,ss.second);
	}
	for(double t = 0; t < bvh->GetMaxTime(); t+=bvh->GetTimeStep())
	{
		int dof = mSkeleton->getPositions().rows();
		Eigen::VectorXd p = Eigen::VectorXd::Zero(dof);
		Eigen::VectorXd p1 = Eigen::VectorXd::Zero(dof);
		//Set p
		bvh->SetMotion(t);
		for(auto ss :mBVHMap)
		{
			dart::dynamics::BodyNode* bn = mSkeleton->getBodyNode(ss.first);
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
		for(auto ss :mBVHMap)
		{
			dart::dynamics::BodyNode* bn = mSkeleton->getBodyNode(ss.first);
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
			v = mSkeleton->getPositionDifferences(p1, p)*20;
		}
		else{
			v = mSkeleton->getPositionDifferences(p, p1)*20;
		}

		for(auto& jn : mSkeleton->getJoints()){
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
		mBVHFrames.push_back(new Frame(p, v));
	}

//calculate contact infomation
	double heightLimit = 0.05;
	double velocityLimit = 6;
	int i = 0;
	for(double t = 0; t < bvh->GetMaxTime(); t+=bvh->GetTimeStep())
	{
		Eigen::VectorXd contact(mContactList.size());
		contact.setZero();

		mSkeleton->setPositions(mBVHFrames[i]->position);
		mSkeleton->setVelocities(mBVHFrames[i]->velocity);
		mSkeleton->computeForwardKinematics(true,true,false);
		
		for(int i = 0; i < mContactList.size(); i++) 
		{
			double height = mSkeleton->getBodyNode(mContactList[i])->getWorldTransform().translation()[1];
			double velocity = mSkeleton->getBodyNode("FootEndR")->getLinearVelocity().norm();
			if(height < heightLimit && velocity < velocityLimit) {
				contact(i) = 1;
			} 

		}
		mBVHFrames[i++]->SetContact(contact);
	}
}

Frame*
Character::
GetTargetPositionsAndVelocitiesFromBVH(BVH* bvh,int t)
{
	return mBVHFrames[t];
}
};