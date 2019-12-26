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
	this->mSkeleton = SkeletonBuilder::BuildFromFile(path);

//	DPhy::SkeletonBuilder::DeformBodyNode(this->mSkeleton, this->mSkeleton->getBodyNode("ArmL"),
//		std::make_tuple("ArmL", 0, 0.5));
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
	mBVHFrames.clear();
	avgCOMVelocity << 0, 0, 0;
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

	for(int i = 0; i < mBVHFrames.size(); i++)
	{
		Eigen::VectorXd contact(mContactList.size());
		contact.setZero();

		mSkeleton->setPositions(mBVHFrames[i]->position);
		mSkeleton->setVelocities(mBVHFrames[i]->velocity);
		mSkeleton->computeForwardKinematics(true,true,false);
		
		for(int j = 0; j < mContactList.size(); j++) 
		{
			double height = mSkeleton->getBodyNode(mContactList[j])->getWorldTransform().translation()[1];
			double velocity = mSkeleton->getBodyNode("FootEndR")->getLinearVelocity().norm();
			if(height < heightLimit && velocity < velocityLimit) {
				contact(j) = 1;
			} 

		}
		mBVHFrames[i]->SetContact(contact);
		mBVHFrames[i]->SetCOMposition(mSkeleton->getRootBodyNode()->getCOM());
		if(i != 0) { 
			mBVHFrames[i]->SetCOMvelocity(mSkeleton->getRootBodyNode()->getCOMLinearVelocity());
			avgCOMVelocity += mBVHFrames[i]->COMvelocity;
		}
	}

	mBVHFrames[0]->SetCOMvelocity(mBVHFrames[1]->COMvelocity);
	avgCOMVelocity += mBVHFrames[0]->COMvelocity;

	totalFrames = mBVHFrames.size();
	avgCOMVelocity /= totalFrames;

	mBVHFrames_r.clear();
}
void
Character::
RescaleOriginalBVH(double w)
{
	mSkeleton->setPositions(mBVHFrames[0]->position);
	mSkeleton->setVelocities(mBVHFrames[0]->velocity);
	mSkeleton->computeForwardKinematics(true,true,false);

	double minheight = 0.0;
	for(int i = 0; i < mContactList.size(); i++) 
	{
		double height = mSkeleton->getBodyNode(mContactList[i])->getWorldTransform().translation()[1];
		if(i == 0 || height < minheight) minheight = height;
	}

	for(int i = 0; i < mBVHFrames.size(); i++)
	{
		Eigen::VectorXd p = mBVHFrames[i]->position;
		p[4] -= minheight - 0.02;
		mBVHFrames[i]->SetPosition(p);
	}

//calculate contact infomation
	double heightLimit = 0.05;
	double velocityLimit = 6;
	Eigen::VectorXd prev_p;
	Eigen::VectorXd prev_v;
	for(int i = 0; i < mBVHFrames.size(); i++)
	{
		if(i != 0) {
			Eigen::VectorXd cur_p = mBVHFrames[i]->position;
			Eigen::Vector3d d_p = cur_p.segment<3>(3) - prev_p.segment<3>(3);
			d_p *= w;
			prev_p = cur_p;
			cur_p.segment<3>(3) = mBVHFrames[i-1]->position.segment<3>(3) + d_p;
			mBVHFrames[i]->SetPosition(cur_p);

			Eigen::VectorXd cur_v = mBVHFrames[i]->velocity;
			cur_v.segment<3>(3) = w * cur_v.segment<3>(3);

			mBVHFrames[i]->SetVelocity(cur_v);

		} else {
			prev_p = mBVHFrames[i]->position;
		}
		Eigen::VectorXd contact(mContactList.size());
		contact.setZero();

		mSkeleton->setPositions(mBVHFrames[i]->position);
		mSkeleton->setVelocities(mBVHFrames[i]->velocity);
		mSkeleton->computeForwardKinematics(true,true,false);
		
		for(int j = 0; j < mContactList.size(); j++) 
		{
			double height = mSkeleton->getBodyNode(mContactList[j])->getWorldTransform().translation()[1];
			double velocity = mSkeleton->getBodyNode("FootEndR")->getLinearVelocity().norm();
			if(height < heightLimit && velocity < velocityLimit) {
				contact(j) = 1;
			} 
		}
		mBVHFrames[i]->SetContact(contact);
		mBVHFrames[i]->SetCOMposition(mSkeleton->getRootBodyNode()->getCOM());
		Eigen::Vector3d v = mSkeleton->getRootBodyNode()->getCOMLinearVelocity();
		mBVHFrames[i]->SetCOMvelocity(v);
	}
}
Frame*
Character::
GetTargetPositionsAndVelocitiesFromBVH(BVH* bvh, double t, bool isPhase)
{
	int bi = 5;
	if(isPhase) {
		int k0 = (int) std::floor(t);
		int k1 = (int) std::ceil(t);	
		if(mBVHFrames_r.size() == 0) {
			for(auto f: mBVHFrames) {
				mBVHFrames_r.push_back(new Frame(f));
			}
		}
		// blending : 이전 클립의 end - 4번째 frame부터 새 클립의 begin + 4 frame까지를 blending
		if(k1 / (mBVHFrames_r.size() - bi) < 1 ) {
			if (k0 == k1)
				return new Frame(mBVHFrames_r[k0]);
			else
				return new Frame(DPhy::BlendPosition(mBVHFrames_r[k1]->position, mBVHFrames_r[k0]->position, (t-k0)), DPhy::BlendPosition(mBVHFrames_r[k1]->velocity, mBVHFrames_r[k0]->velocity, (t-k0)));		
		}
		else {
			Eigen::Vector6d root_next = mBVHFrames[0]->position.segment<6>(0);
			Eigen::Vector6d root_prev = mBVHFrames_r[mBVHFrames_r.size() - 1]->position.segment<6>(0);

			Eigen::AngleAxisd root_next_ori(root_next.segment<3>(0).norm(), root_next.segment<3>(0).normalized());
			Eigen::AngleAxisd root_prev_ori(root_prev.segment<3>(0).norm(), root_prev.segment<3>(0).normalized());

			Eigen::Matrix3d root_dori;
			root_dori = root_prev_ori * root_next_ori.inverse();
			root_dori = DPhy::projectToXZ(root_dori);

			std::vector<Eigen::VectorXd> positions;

			int prev_size = mBVHFrames_r.size();
			for(int i = 0; i < mBVHFrames.size(); i++) {
				Eigen::VectorXd position_next = mBVHFrames[i]->position;

				Eigen::Vector3d dpos = mBVHFrames[i]->position.segment<3>(3) - mBVHFrames[0]->position.segment<3>(3);
				dpos =  root_dori * dpos + root_prev.segment<3>(3);
				dpos[1] = position_next[4];

				Eigen::AngleAxisd cur_ori(mBVHFrames[i]->position.segment<3>(0).norm(), mBVHFrames[i]->position.segment<3>(0).normalized());
				Eigen::Matrix3d dori;
				dori = root_dori * cur_ori;

				Eigen::Quaterniond dori_q(dori);

				position_next.segment<3>(3) = dpos;
				position_next.segment<3>(0) = DPhy::QuaternionToDARTPosition(dori_q);

				positions.push_back(position_next);
				mBVHFrames_r.push_back(new Frame(positions[i], mBVHFrames[i]->velocity));

			}
			for(int i = 0; i < bi; i++) {
				int idx = prev_size - (i + 1);
				Eigen::VectorXd position_prev = mBVHFrames_r[idx]->position;
				double weight = 1.0 - (i+1) / (double)(bi+1);
				mBVHFrames_r[idx]->position = DPhy::BlendPosition(positions[0], position_prev, weight);
				mBVHFrames_r[idx]->velocity = mSkeleton->getPositionDifferences(mBVHFrames_r[idx-1]->position, mBVHFrames_r[idx]->position)* 1.0 / bvh->GetTimeStep();
					// position_next[4] = weight * position_next[4] + (1 - weight) * position_prev[4]; 

			}
			if (k0 == k1)
				return new Frame(mBVHFrames_r[k0]);
			else
				return new Frame(DPhy::BlendPosition(mBVHFrames_r[k1]->position, mBVHFrames_r[k0]->position, (t-k0)), 
					DPhy::BlendPosition(mBVHFrames_r[k1]->velocity, mBVHFrames_r[k0]->velocity, (t-k0)));	
		}	
	}
	else {
		int k0 = (int) std::min(std::floor(t), (double)mBVHFrames.size()-1);
		int k1 = (int) std::min(std::ceil(t), (double)mBVHFrames.size()-1);
		if (k0 == k1) 
			return new Frame(mBVHFrames[k0]);

		Frame* k0_f = mBVHFrames[k0];
		Frame* k1_f = mBVHFrames[k1];
		
		double w = t - k0;
		int size = k0_f->position.size();
		Eigen::VectorXd position(size);
		Eigen::VectorXd velocity(size);

		for(int i = 0; i < size; i += 3) {
			if(i == 3) {
				position.segment<3>(i) = (1 - w) * k0_f->position.segment<3>(i) + w * k1_f->position.segment<3>(i);
				velocity.segment<3>(i) = (1 - w) * k0_f->velocity.segment<3>(i) + w * k1_f->velocity.segment<3>(i);
			} else {
				Eigen::Quaterniond k0_q = DPhy::DARTPositionToQuaternion(k0_f->position.segment<3>(i));
				Eigen::Quaterniond k1_q = DPhy::DARTPositionToQuaternion(k1_f->position.segment<3>(i));
				position.segment<3>(i) = DPhy::QuaternionToDARTPosition(k0_q.slerp(w, k1_q));

				k0_q = DPhy::DARTPositionToQuaternion(k0_f->velocity.segment<3>(i));
				k1_q = DPhy::DARTPositionToQuaternion(k1_f->velocity.segment<3>(i));
				velocity.segment<3>(i) = DPhy::QuaternionToDARTPosition(k0_q.slerp(w, k1_q));
			}
		}

		double heightLimit = 0.05;
		double velocityLimit = 6;

		Eigen::VectorXd contact(mContactList.size());
		contact.setZero();

		mSkeleton->setPositions(position);
		mSkeleton->setVelocities(velocity);
		mSkeleton->computeForwardKinematics(true,true,false);
			
		for(int j = 0; j < mContactList.size(); j++) 
		{
			double height = mSkeleton->getBodyNode(mContactList[j])->getWorldTransform().translation()[1];
			double velocity = mSkeleton->getBodyNode("FootEndR")->getLinearVelocity().norm();
			if(height < heightLimit && velocity < velocityLimit) {
				contact(j) = 1;
			}
		}
		Frame* newFrame =  new Frame(position, velocity, contact);
		newFrame->SetCOMposition(mSkeleton->getRootBodyNode()->getCOM());
		newFrame->SetCOMvelocity(w * k1_f->COMvelocity + (1 - w) * k0_f->COMvelocity);
		return newFrame;
	}
}
void
Character::
EditTrajectory(BVH* bvh, int t, double w) {
	int count = 0;
	for(int i = t; i < mBVHFrames.size(); i++) {
		if(mBVHFrames[i+1]->COMvelocity[1] > mBVHFrames[i]->COMvelocity[1]) break;
		count++;
	}
	double slope = (mBVHFrames[t]->COMvelocity[1] - mBVHFrames[t+count]->COMvelocity[1]) * 30.0 / count;
	double climax_dt = mBVHFrames[t]->COMvelocity[1] / slope * 30.0;
	double dheight = 0.5 * slope * (climax_dt / 30.0) * (climax_dt / 30.0);
	double targetHeight = mBVHFrames[t]->COMposition[1] + dheight * w;
	double new_climax_dt = sqrt(1 / (0.5 * slope) * dheight * w) * 30.0;
	double new_com_velocity = new_climax_dt * slope / 30.0;

	double w0 = new_com_velocity / mBVHFrames[t]->COMvelocity[1];
	
	std::vector<Frame*> mBVHFrames_;

	for(int i = 0; i < t; i++) {
		Frame* f = new Frame(mBVHFrames[i]); 

		f->COMvelocity[1] *= w0;
		f->velocity[4] *= w0;

		if(i != 0) {
			f->position[4] = mBVHFrames_[mBVHFrames_.size()-1]->position[4] + f->COMvelocity[1] / 30.0;
		}
		mBVHFrames_.push_back(f);

	}
	double dt_scale = std::floor(climax_dt) / std::floor(new_climax_dt);
	for(int i = t; i < t + std::floor(new_climax_dt) * 2; i++) {
		Frame* f = GetTargetPositionsAndVelocitiesFromBVH(bvh, t + (i-t) * dt_scale);
		f->COMvelocity[1] = new_com_velocity - (i-t) * slope / 30.0;
		f->position[4] = mBVHFrames_[mBVHFrames_.size()-1]->position[4] + f->COMvelocity[1] / 30.0;
		mBVHFrames_.push_back(f);

	}
	w0 = abs(new_com_velocity / mBVHFrames[ t + std::floor(climax_dt) * 2]->COMvelocity[1]);

	for(int i = t + std::floor(climax_dt) * 2; i < mBVHFrames.size(); i++) {
		Frame* f = new Frame(mBVHFrames[i]); 

		f->COMvelocity[1] *= w0;
		f->velocity[4] *= w0;
		f->position[4] = mBVHFrames_[mBVHFrames_.size()-1]->position[4] + f->COMvelocity[1] / 30.0;

		mBVHFrames_.push_back(f);
	}
	
	Eigen::VectorXd position, velocity;
	std::vector<Eigen::Vector3d> pos_ori;
	for(int i = 0; i < t; i++) {
		pos_ori.clear();
		position = mBVHFrames[i]->position;
		velocity = mBVHFrames[i]->velocity;

		mSkeleton->setPositions(position);
		mSkeleton->setVelocities(velocity);
		mSkeleton->computeForwardKinematics(true,true,false);

		pos_ori.push_back(mSkeleton->getBodyNode("FootL")->getTransform().translation());
		pos_ori.push_back(mSkeleton->getBodyNode("FootR")->getTransform().translation());
		position = mBVHFrames_[i]->position;
		velocity = mBVHFrames_[i]->velocity;

		mSkeleton->setPositions(position);
		mSkeleton->setVelocities(velocity);
		mSkeleton->computeForwardKinematics(true,true,false);
		pos_ori.push_back(mSkeleton->getBodyNode("Spine")->getTransform().translation());

		std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> constraints;
		constraints.push_back(std::make_tuple("FootL", pos_ori[0], Eigen::Vector3d(0, 0, 0)));
		constraints.push_back(std::make_tuple("FootR", pos_ori[1], Eigen::Vector3d(0, 0, 0)));
		constraints.push_back(std::make_tuple("Spine", pos_ori[2], Eigen::Vector3d(0, 0, 0)));

		Eigen::VectorXd new_position = DPhy::solveMCIK(mSkeleton, constraints);
		mBVHFrames_[i]->position = new_position;
		mBVHFrames_[i]->COMposition = mSkeleton->getCOM();

		if(i != 0) {
			mBVHFrames_[i]->velocity = mSkeleton->getPositionDifferences(mBVHFrames_[i]->position, mBVHFrames_[i-1]->position) * 20;
			mBVHFrames_[i]->COMvelocity = (mBVHFrames_[i]->COMposition - mBVHFrames_[i-1]->COMposition) * 20;
		}

	}
	for(int i = t + std::floor(new_climax_dt) * 2, j = t + std::floor(climax_dt) * 2; j < mBVHFrames.size(); i++, j++) {
		pos_ori.clear();
		position = mBVHFrames[j]->position;
		velocity = mBVHFrames[j]->velocity;

		mSkeleton->setPositions(position);
		mSkeleton->setVelocities(velocity);
		mSkeleton->computeForwardKinematics(true,true,false);

		pos_ori.push_back(mSkeleton->getBodyNode("FootL")->getTransform().translation());
		pos_ori.push_back(mSkeleton->getBodyNode("FootR")->getTransform().translation());
		position = mBVHFrames_[i]->position;
		velocity = mBVHFrames_[i]->velocity;

		mSkeleton->setPositions(position);
		mSkeleton->setVelocities(velocity);
		mSkeleton->computeForwardKinematics(true,true,false);
		pos_ori.push_back(mSkeleton->getBodyNode("Spine")->getTransform().translation());

		std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> constraints;
		constraints.push_back(std::make_tuple("FootL", pos_ori[0], Eigen::Vector3d(0, 0, 0)));
		constraints.push_back(std::make_tuple("FootR", pos_ori[1], Eigen::Vector3d(0, 0, 0)));
		constraints.push_back(std::make_tuple("Spine", pos_ori[2], Eigen::Vector3d(0, 0, 0)));

		Eigen::VectorXd new_position = DPhy::solveMCIK(mSkeleton, constraints);
		mBVHFrames_[i]->position = new_position;
		mBVHFrames_[i]->COMposition = mSkeleton->getCOM();
		if(i != 0) {
			mBVHFrames_[i]->velocity = mSkeleton->getPositionDifferences(mBVHFrames_[i]->position, mBVHFrames_[i-1]->position) * 20;
			mBVHFrames_[i]->COMvelocity = (mBVHFrames_[i]->COMposition - mBVHFrames_[i-1]->COMposition) * 20;
		}
	}
	mBVHFrames = mBVHFrames_;

}
};
