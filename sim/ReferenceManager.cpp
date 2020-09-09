#include "ReferenceManager.h"
#include <tinyxml.h>
#include <fstream>
#include <stdlib.h>
#include <cmath>

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


	auto& skel = mCharacter->GetSkeleton();
	mDOF = skel->getPositions().rows();
}
void
ReferenceManager::
ComputeAxisMean(){
	mAxis_BVH.clear();
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < mPhaseLength - 1; i++) {
		Eigen::VectorXd m_cur = mMotions_phase[i]->GetPosition();
		Eigen::VectorXd m_next = mMotions_phase[i+1]->GetPosition();

		Eigen::VectorXd axis(mDOF);
		for(int j = 0; j < n_bnodes; j++) {
			int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);

			if(dof == 6) {
				axis.segment<3>(idx) = JointPositionDifferences(m_next.segment<3>(idx), m_cur.segment<3>(idx));

				Eigen::AngleAxisd root_ori = Eigen::AngleAxisd(m_cur.segment<3>(idx).norm(), m_cur.segment<3>(idx).normalized());
				Eigen::Vector3d v = m_next.segment<3>(idx + 3) - m_cur.segment<3>(idx + 3);
				axis.segment<3>(idx + 3) = root_ori.inverse() * v;
			} else if(dof == 3) {
				axis.segment<3>(idx) = JointPositionDifferences(m_next.segment<3>(idx), m_cur.segment<3>(idx));
			} else {
				axis(idx) = m_next(idx) - m_cur(idx);
			}
		}
		mAxis_BVH.push_back(axis);
	}
	mAxis_BVH.push_back(mAxis_BVH[0]);
}
void
ReferenceManager::
ComputeAxisDev() {
	mDev_BVH.clear();
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < mPhaseLength; i++) {
		int t = i + mPhaseLength;
		std::vector<std::pair<Eigen::VectorXd, double>> data;
		data.clear();
		for(int j = t - 3; j <= t + 3; j++) {
			int t_ = j % mPhaseLength;
			Eigen::VectorXd y(mDOF);
			y.setZero();

			for(int k = 0; k < n_bnodes; k++) {
				int dof = mCharacter->GetSkeleton()->getBodyNode(k)->getParentJoint()->getNumDofs();
				int idx = mCharacter->GetSkeleton()->getBodyNode(k)->getParentJoint()->getIndexInSkeleton(0);

				if(dof == 6) {
					Eigen::Vector3d diff = mAxis_BVH[i].segment<3>(idx) - mAxis_BVH[t_].segment<3>(idx);
					double x = diff.dot(mAxis_BVH[i].segment<3>(idx).normalized());
 					y(idx) = (diff - x * mAxis_BVH[i].segment<3>(idx).normalized()).norm() 
 							/ std::max(mAxis_BVH[i].segment<3>(idx).norm(), 0.02);	
				
					diff = mAxis_BVH[i].segment<3>(idx + 3) - mAxis_BVH[t_].segment<3>(idx + 3);
					x = diff.dot(mAxis_BVH[i].segment<3>(idx + 3).normalized());
 					y(idx + 3) = (diff - x * mAxis_BVH[i].segment<3>(idx + 3).normalized()).norm() 
 							/ std::max(mAxis_BVH[i].segment<3>(idx + 3).norm(), 0.02);

				} else if(dof == 3) {
					Eigen::Vector3d diff  = mAxis_BVH[i].segment<3>(idx) - mAxis_BVH[t_].segment<3>(idx);
					double x = diff.dot(mAxis_BVH[i].segment<3>(idx).normalized());
 					y(idx) = (diff - x * mAxis_BVH[i].segment<3>(idx).normalized()).norm() 
 							/ std::max(mAxis_BVH[i].segment<3>(idx).norm(), 0.02);				
 				} else {
					y(idx) = 0;
				}
				
			}
 			data.push_back(std::pair<Eigen::VectorXd, double>(y, (1 - abs(t - j) * 0.3)));
		}
	 	Eigen::VectorXd dev(mDOF);
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
SaveAdaptiveMotion(std::string postfix) {
	std::string path = mPath + std::string("adaptive") + postfix;
	std::cout << "save motion to:" << path << std::endl;

	std::ofstream ofs(path);

	for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
		ofs << mMotions_phase_adaptive[i]->GetPosition().transpose() << std::endl;
		ofs << mMotions_phase_adaptive[i]->GetVelocity().transpose() << std::endl;
	}
	ofs.close();
	
	path = mPath + std::string("cp") + postfix;
	ofs.open(path);
	ofs << mKnots.size() << std::endl;
	for(auto t: mKnots) {	
		ofs << t << std::endl;
	}
		
	for(auto t: mPrevCps) {	
		ofs << t.transpose() << std::endl;
	}
	ofs.close();
}
void 
ReferenceManager::
LoadAdaptiveMotion(std::vector<Eigen::VectorXd> cps) {
	DPhy::MultilevelSpline* s = new DPhy::MultilevelSpline(1, mPhaseLength);

	s->SetKnots(0, mKnots);
	s->SetControlPoints(0, cps);

	std::vector<Eigen::VectorXd> newpos;
	std::vector<Eigen::VectorXd> new_displacement = s->ConvertSplineToMotion();
	this->AddDisplacementToBVH(new_displacement, newpos);
	std::vector<Eigen::VectorXd> newvel = this->GetVelocityFromPositions(newpos);

	for(int j = 0; j < mPhaseLength; j++) {
		mMotions_phase_adaptive[j]->SetPosition(newpos[j]);
		mMotions_phase_adaptive[j]->SetVelocity(newvel[j]);
	}

	this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase_adaptive, mMotions_gen_adaptive);

}
void 
ReferenceManager::
LoadAdaptiveMotion(std::string postfix) {
	std::string path = mPath + std::string("adaptive") + postfix;

	std::ifstream is(path);
	if(is.fail())
		return;
	std::cout << "load Motion from: " << path << std::endl;

	char buffer[256];
	for(int i = 0; i < mPhaseLength; i++) {
		Eigen::VectorXd pos(mDOF);
		Eigen::VectorXd vel(mDOF);
		for(int j = 0; j < mDOF; j++) 
		{
			is >> buffer;
			pos[j] = atof(buffer);
		}
		for(int j = 0; j < mDOF; j++) 
		{
			is >> buffer;
			vel[j] = atof(buffer);
		}
		mMotions_phase_adaptive[i]->SetPosition(pos);
		mMotions_phase_adaptive[i]->SetVelocity(vel);
	}
	is.close();
	
	path = mPath + std::string("cp") + postfix;
	is.open(path);
	mKnots.clear();
	is >> buffer;
	int knot_size = atoi(buffer);
	for(int i = 0; i < knot_size; i++) {	
		is >> buffer;
		mKnots.push_back(atoi(buffer));
	}
	for(int i = 0; i < knot_size; i++) {	
		Eigen::VectorXd cps(mDOF);	
		for(int j = 0; j < mDOF; j++) {
			is >> buffer;
			cps[j] = atof(buffer);
		}
		mPrevCps[i] = cps;
	}
	is.close();

	this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase_adaptive, mMotions_gen_adaptive);

	// std::vector<std::pair<Eigen::VectorXd,double>> pos;
	// for(int i = 0; i < mPhaseLength; i++) {
	// 	pos.push_back(std::pair<Eigen::VectorXd,double>(mMotions_phase_adaptive[i]->GetPosition(), i));
	// }
	// MultilevelSpline* s = new MultilevelSpline(1, mPhaseLength);
	// s->SetKnots(0, mKnots);

	// s->ConvertMotionToSpline(pos);
	// path = std::string(CAR_DIR) + std::string("/result/jump_spline_ref");
	
	// std::vector<Eigen::VectorXd> cps = s->GetControlPoints(0);

	// std::ofstream ofs(path);

	// ofs << mKnots.size() << std::endl;
	// for(auto t: mKnots) {	
	// 	ofs << t << std::endl;
	// }
	// for(auto t: cps) {	
	// 	ofs << t.transpose() << std::endl;
	// }

	// ofs << pos.size() << std::endl;
	// for(auto t: pos) {	
	// 	ofs << t.second << std::endl;
	// 	ofs << t.first.transpose() << std::endl;
	// }
	// ofs.close();

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

	std::vector<std::string> contact;
	contact.clear();
	contact.push_back("FootEndR");
	contact.push_back("FootR");
	contact.push_back("FootEndL");
	contact.push_back("FootL");

	auto& skel = mCharacter->GetSkeleton();
	int dof = skel->getPositions().rows();
	std::map<std::string,std::string> bvhMap = mCharacter->GetBVHMap(); 
	for(const auto ss :bvhMap){
		bvh->AddMapping(ss.first,ss.second);
	}


	double t = 0;
	for(int i = 0; i < bvh->GetMaxFrame(); i++)
	{
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
				if(ss.first.find("Arm") != std::string::npos)
					p[jn->getIndexInSkeleton(0)] = a[1];
				else	
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
		
		auto& skel = this->mCharacter->GetSkeleton();
	
		skel->setPositions(p);
		skel->computeForwardKinematics(true,false,false);

		std::vector<bool> c;
		for(int j = 0; j < contact.size(); j++) {
			Eigen::Vector3d p = skel->getBodyNode(contact[j])->getWorldTransform().translation();
			c.push_back(p[1] < 0.04);
		}
		mContacts.push_back(c);

		t += bvh->GetTimeStep();
	}
	mMotions_raw.back()->SetVelocity(mMotions_raw.front()->GetVelocity());

	mPhaseLength = mMotions_raw.size();
	mTimeStep = bvh->GetTimeStep();

	for(int i = 0; i < mPhaseLength; i++) {
		mMotions_phase.push_back(new Motion(mMotions_raw[i]));
		if(i != 0 && i != mPhaseLength - 1) {
			for(int j = 0; j < contact.size(); j++)
				if(mContacts[i-1][j] && mContacts[i+1][j] && !mContacts[i][j])
						mContacts[i][j] = true;
		}
	 }

	delete bvh;
	this->ComputeAxisMean();
	this->ComputeAxisDev();
	this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase, mMotions_gen);

}
std::vector<Eigen::VectorXd> 
ReferenceManager::
GetVelocityFromPositions(std::vector<Eigen::VectorXd> pos)
{
	std::vector<Eigen::VectorXd> vel;
	auto skel = mCharacter->GetSkeleton();
	for(int i = 0; i < pos.size() - 1; i++) {
		Eigen::VectorXd v = skel->getPositionDifferences(pos[i + 1], pos[i]) / 0.033;
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
		vel.push_back(v);
	}
	vel.push_back(vel.front());

	return vel;
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
void ReferenceManager::GenerateMotionsFromSinglePhase(int frames, bool blend, std::vector<Motion*>& p_phase, std::vector<Motion*>& p_gen)
{
	mLock.lock();
	while(!p_gen.empty()){
		Motion* m = p_gen.back();
		p_gen.pop_back();

		delete m;
	}		

	auto& skel = mCharacter->GetSkeleton();

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	
	skel->setPositions(p_phase[0]->GetPosition());
	skel->computeForwardKinematics(true,false,false);

	Eigen::Vector3d p0_footl = skel->getBodyNode("FootL")->getWorldTransform().translation();
	Eigen::Vector3d p0_footr = skel->getBodyNode("FootR")->getWorldTransform().translation();


	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform(p_phase[0]->GetPosition().head<6>());
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform(p_phase.back()->GetPosition().head<6>());

	Eigen::Isometry3d T0_gen = T0_phase;
	
	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();

	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());			
	T01.linear() = dart::math::expMapRot(DPhy::projectToXZ(p01));
	T01.translation()[1] = 0;

	for(int i = 0; i < frames; i++) {
		
		int phase = i % mPhaseLength;
		
		if(i < mPhaseLength) {
			p_gen.push_back(new Motion(p_phase[i]));
		} else {
			Eigen::VectorXd pos;
			if(phase == 0) {
				std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> constraints;
	
				skel->setPositions(p_gen.back()->GetPosition());
				skel->computeForwardKinematics(true,false,false);

				Eigen::Vector3d p_footl = skel->getBodyNode("FootL")->getWorldTransform().translation();
				Eigen::Vector3d p_footr = skel->getBodyNode("FootR")->getWorldTransform().translation();

				p_footl(1) = p0_footl(1);
				p_footr(1)= p0_footr(1);

				constraints.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>("FootL", p_footl, Eigen::Vector3d(0, 0, 0)));
				constraints.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>("FootR", p_footr, Eigen::Vector3d(0, 0, 0)));

				Eigen::VectorXd p = p_phase[phase]->GetPosition();
				p.segment<3>(3) = p_gen.back()->GetPosition().segment<3>(3);

				skel->setPositions(p);
				skel->computeForwardKinematics(true,false,false);
				pos = solveMCIKRoot(skel, constraints);
				pos(4) = p_phase[phase]->GetPosition()(4);
				
				T0_gen = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
			} else {
				pos = p_phase[phase]->GetPosition();
				Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
				T_current = T0_phase.inverse()*T_current;
				T_current = T0_gen*T_current;
				pos.head<6>() = dart::dynamics::FreeJoint::convertToPositions(T_current);
			}

			Eigen::VectorXd vel = skel->getPositionDifferences(pos, p_gen.back()->GetPosition()) / 0.033;
			p_gen.back()->SetVelocity(vel);
			p_gen.push_back(new Motion(pos, vel));

			if(blend && phase == 0) {
				for(int j = mBlendingInterval; j > 0; j--) {
					double weight = 1.0 - j / (double)(mBlendingInterval+1);
					Eigen::VectorXd oldPos = p_gen[i - j]->GetPosition();
					p_gen[i - j]->SetPosition(DPhy::BlendPosition(oldPos, pos, weight));
					vel = skel->getPositionDifferences(p_gen[i - j]->GetPosition(), p_gen[i - j - 1]->GetPosition()) / 0.033;
			 		p_gen[i - j - 1]->SetVelocity(vel);
				}
			}
		}
	}
	mLock.unlock();

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
Eigen::VectorXd 
ReferenceManager::
GetAxisMean(double t) {
	int k0 = (int) std::floor(t);
	if(k0 == mPhaseLength)
		k0 = 0;
	return mAxis_BVH[k0];
}
Eigen::VectorXd 
ReferenceManager::
GetAxisDev(double t) {
	int k0 = (int) std::floor(t);
	if(k0 == mPhaseLength)
		k0 = 0;
	return mDev_BVH[k0];
}
void
ReferenceManager::
InitOptimization(int nslaves, std::string save_path) {
	mKnots.push_back(0);
	mKnots.push_back(12);
	mKnots.push_back(29);
	mKnots.push_back(37);
	mKnots.push_back(44);
	mKnots.push_back(52);
	mKnots.push_back(56);
	mKnots.push_back(59);
	mKnots.push_back(64);
	mKnots.push_back(76);

	for(int i = 0; i < this->mKnots.size() + 3; i++) {
		mPrevCps.push_back(Eigen::VectorXd::Zero(mDOF));
	}
	for(int i = 0; i < this->GetPhaseLength(); i++) {
		mMotions_phase_adaptive.push_back(new Motion(mMotions_phase[i]));
	}
	this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase_adaptive, mMotions_gen_adaptive);
	for(int i = 0; i < nslaves; i++) {
		std::vector<Motion*> mlist;
		mMotions_gen_temp.push_back(mlist);
	}

	nOp = 0;
	mPath = save_path;
	mPrevRewardTrajectory = 0.9;
	mPrevRewardTarget = 0.05;	
	
	for(int i = 0; i < 3; i++) {
		nRejectedSamples.push_back(0);
	}
	mRefUpdateMode = true;
	mTargetRefUpdate = 1.25;
	// std::vector<std::pair<Eigen::VectorXd,double>> pos;
	// for(int i = 0; i < mPhaseLength; i++) {
	// 	pos.push_back(std::pair<Eigen::VectorXd,double>(mMotions_phase[i]->GetPosition(), i));
	// }
	// MultilevelSpline* s = new MultilevelSpline(1, mPhaseLength);
	// s->SetKnots(0, mKnots);

	// s->ConvertMotionToSpline(pos);
	// std::string path = std::string(CAR_DIR) + std::string("/result/walk_base");
	
	// std::vector<Eigen::VectorXd> cps = s->GetControlPoints(0);

	// std::ofstream ofs(path);

	// ofs << mKnots.size() << std::endl;
	// for(auto t: mKnots) {	
	// 	ofs << t << std::endl;
	// }
	// for(int i = 0; i < cps.size(); i++) {
	// 	if(i >= 1 && i <= cps.size() - 3)	
	// 		ofs << cps[i].transpose() << std::endl;
	// }

	// ofs << pos.size() << std::endl;
	// for(auto t: pos) {	
	// 	ofs << t.second << std::endl;
	// 	ofs << t.first.transpose() << std::endl;
	// }
	// ofs.close();

}
std::vector<double> 
ReferenceManager::
GetContacts(double t)
{
	std::vector<double> result;
	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	

	if (k0 == k1) {
		int phase = k0 % mPhaseLength;
		std::vector<bool> contact = mContacts[phase];
		for(int i = 0; i < contact.size(); i++)
			result.push_back(contact[i]);
	} else {
		int phase0 = k0 % mPhaseLength;
		int phase1 = k1 % mPhaseLength;

		std::vector<bool> contact0 = mContacts[phase0];
		std::vector<bool> contact1 = mContacts[phase1];
		for(int i = 0; i < contact0.size(); i++) {
			if(contact0[i] == contact1[i])
				result.push_back(contact0[i]);
			else 
				result.push_back(0.5);
		}

	}
	return result;
}
std::vector<std::pair<bool, Eigen::Vector3d>> 
ReferenceManager::
GetContactInfo(Eigen::VectorXd pos) 
{
	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	
	skel->setPositions(pos);
	skel->computeForwardKinematics(true,false,false);

	std::vector<std::string> contact;
	contact.clear();
	contact.push_back("FootEndR");
	contact.push_back("FootR");
	contact.push_back("FootEndL");
	contact.push_back("FootL");

	std::vector<std::pair<bool, Eigen::Vector3d>> result;
	result.clear();
	for(int i = 0; i < contact.size(); i++) {
		Eigen::Vector3d p = skel->getBodyNode(contact[i])->getWorldTransform().translation();
		if(p[1] < 0.07) {
			result.push_back(std::pair<bool, Eigen::Vector3d>(true, p));
		} else {
			result.push_back(std::pair<bool, Eigen::Vector3d>(false, p));
		}
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	return result;
}
void 
ReferenceManager::
SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, 
				 std::pair<double, double> rewards,
				 Eigen::VectorXd parameters) {
	
	std::vector<int> flag;
	if((rewards.first / mPhaseLength)  < 0.9)
		flag.push_back(0);
	else
		flag.push_back(1);

	if(rewards.second < mPrevRewardTarget)
		flag.push_back(0);
	else
		flag.push_back(1);

	if(flag[0] == 0)
		return;

	// if((rewards.first / mPhaseLength)  < 0.9 || rewards.second < mPrevRewardTarget) {
	// 	nRejectedSamples[0] += 1;
	// 	if ((rewards.first / mPhaseLength) >= 0.9 && rewards.second < mPrevRewardTarget) {
	// 		nRejectedSamples[1] += 1;
	// 	} else if((rewards.first / mPhaseLength) < 0.9 && rewards.second < mPrevRewardTarget) {
	// 		nRejectedSamples[2] += 1;
	// 	}
	// 	return;
	// }

	MultilevelSpline* s = new MultilevelSpline(1, this->GetPhaseLength());
	s->SetKnots(0, mKnots);

	double start_phase = std::fmod(data_spline[0].second, mPhaseLength);
	std::vector<Eigen::VectorXd> trajectory;
	for(int i = 0; i < data_spline.size(); i++) {
		trajectory.push_back(data_spline[i].first);
	}
	trajectory = Align(trajectory, this->GetPosition(start_phase).segment<6>(0));

	std::vector<std::pair<Eigen::VectorXd,double>> displacement;
	for(int i = 0; i < data_spline.size(); i++) {
		data_spline[i].first = trajectory[i];
	}
	this->GetDisplacementWithBVH(data_spline, displacement);
	s->ConvertMotionToSpline(displacement);

	std::vector<Eigen::VectorXd> newpos;
	std::vector<Eigen::VectorXd> new_displacement = s->ConvertSplineToMotion();
	this->AddDisplacementToBVH(new_displacement, newpos);


	double r_slide = 0;
	std::vector<std::vector<std::pair<bool, Eigen::Vector3d>>> c;
	for(int i = 0; i < newpos.size(); i++) {
		c.push_back(this->GetContactInfo(newpos[i]));
	}
	for(int i = 1; i < newpos.size(); i++) {
		if(i < newpos.size() - 1) {
			for(int j = 0; j < 4; j++) {
				if((c[i-1][j].first) && (c[i+1][j].first) && !(c[i][j].first)) 
					(c[i][j].first) = true;
			}
		}
		for(int j = 0; j < 2; j++) {
			bool c_prev_j = (c[i-1][2*j].first) && (c[i-1][2*j + 1].first);
			bool c_cur_j = (c[i][2*j].first) && (c[i][2*j + 1].first);
			if(c_prev_j && c_cur_j) {
				double d = ((c[i-1][2*j].second + c[i-1][2*j+1].second) - (c[i][2*j].second + c[i][2*j+1].second)).norm()*0.5; 
				r_slide += pow(d*4, 2);
			} 
		}
	}
	r_slide = exp(-r_slide);
	auto cps = s->GetControlPoints(0);
	double r_regul = 0;
	for(int i = 0; i < cps.size(); i++) {
		r_regul += cps[i].norm();	
	}
	double reward_trajectory = 0.4 * exp(-pow(r_regul, 2)*0.01) + 0.6 * r_slide;
	mLock.lock();

	if(r_slide > 0.86)
		mRegressionSamples.push_back(std::pair<std::vector<Eigen::VectorXd>, Eigen::VectorXd>(cps, parameters));
	if(flag[1] && mRefUpdateMode) {
		mSamples.push_back(std::tuple<MultilevelSpline*, std::pair<double, double>,  double>(s, 
							std::pair<double, double>(reward_trajectory, r_slide), rewards.second));
		mSampleTargets.push_back(parameters);
		std::string path = mPath + std::string("samples") + std::to_string(nOp);

		std::ofstream ofs;
		ofs.open(path, std::fstream::out | std::fstream::app);

		for(auto t: data_spline) {	
			ofs << t.first.transpose() << " " << t.second << " " << r_slide << std::endl;
		}
		ofs.close();
	}

	mLock.unlock();


}
bool cmp(const std::tuple<DPhy::MultilevelSpline*, std::pair<double, double>, double> &p1, const std::tuple<DPhy::MultilevelSpline*, std::pair<double, double>, double> &p2){
    if(std::get<1>(p1) > std::get<1>(p2)){
        return true;
    }
    else{
        return false;
    }
}
void 
ReferenceManager::
AddDisplacementToBVH(std::vector<Eigen::VectorXd> displacement, std::vector<Eigen::VectorXd>& position) {
	position.clear();
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < displacement.size(); i++) {

		Eigen::VectorXd p_bvh = mMotions_phase[i]->GetPosition();
		Eigen::VectorXd d = displacement[i];
		Eigen::VectorXd p(mCharacter->GetSkeleton()->getNumDofs());

		for(int j = 0; j < n_bnodes; j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			if(dof == 6) {
				p.segment<3>(idx) = Rotate3dVector(p_bvh.segment<3>(idx), d.segment<3>(idx));
				p.segment<3>(idx + 3) = d.segment<3>(idx + 3) + p_bvh.segment<3>(idx + 3);
			} else if (dof == 3) {
				p.segment<3>(idx) = Rotate3dVector(p_bvh.segment<3>(idx), d.segment<3>(idx));
			} else {
				p(idx) = d(idx) + p_bvh(idx);
			}
		}
		position.push_back(p);
	}
}
void
ReferenceManager::
GetDisplacementWithBVH(std::vector<std::pair<Eigen::VectorXd, double>> position, std::vector<std::pair<Eigen::VectorXd, double>>& displacement) {
	displacement.clear();
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	for(int i = 0; i < position.size(); i++) {
		double phase = std::fmod(position[i].second, mPhaseLength);
		
		Eigen::VectorXd p = position[i].first;
		Eigen::VectorXd p_bvh = this->GetPosition(phase);
		Eigen::VectorXd d(mCharacter->GetSkeleton()->getNumDofs());
		for(int j = 0; j < n_bnodes; j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			
			if(dof == 6) {
				d.segment<3>(idx) = JointPositionDifferences(p.segment<3>(idx), p_bvh.segment<3>(idx));
				d.segment<3>(idx + 3) = p.segment<3>(idx + 3) -  p_bvh.segment<3>(idx + 3);
			} else if (dof == 3) {
				d.segment<3>(idx) = JointPositionDifferences(p.segment<3>(idx), p_bvh.segment<3>(idx));
			} else {
				d(idx) = p(idx) - p_bvh(idx);
			}
		}
		displacement.push_back(std::pair<Eigen::VectorXd,double>(d, phase));
	}
}
bool 
ReferenceManager::
Optimize() {
	if(!mRefUpdateMode)
		return false;

	double rewardTarget = 0;
	double rewardTrajectory = 0;
    int mu = 60;
    std::cout << "num sample: " << mSamples.size() << std::endl;
    if(mSamples.size() < 300) {
    	for(int i = 0; i < nRejectedSamples.size(); i++) {
			std::cout << i << " " << nRejectedSamples[i] << std::endl;
		}
    	return false;
    }

    std::stable_sort(mSamples.begin(), mSamples.end(), cmp);
	MultilevelSpline* mean_spline = new MultilevelSpline(1, this->GetPhaseLength()); 
	mean_spline->SetKnots(0, mKnots);
	

	std::vector<Eigen::VectorXd> mean_cps;   
   	mean_cps.clear();
   	int num_knot = mean_spline->GetKnots(0).size();
   	for(int i = 0; i < num_knot + 3; i++) {
		mean_cps.push_back(Eigen::VectorXd::Zero(mDOF));
	}
	double weight_sum = 0;

	std::string path = mPath + std::string("rewards");
	std::ofstream ofs;
	ofs.open(path, std::fstream::out | std::fstream::app);

	for(int i = 0; i < mu; i++) {
		double w = log(mu + 1) - log(i + 1);
	    weight_sum += w;
	    std::vector<Eigen::VectorXd> cps = std::get<0>(mSamples[i])->GetControlPoints(0);
	    for(int j = 0; j < num_knot + 3; j++) {
			mean_cps[j] += w * cps[j];
	    }
	    rewardTrajectory += w * std::get<1>(mSamples[i]).first;
	    rewardTarget += std::get<2>(mSamples[i]);
	    ofs << std::get<1>(mSamples[i]).second << " ";

	}
	ofs << std::endl;
	ofs.close();

	rewardTrajectory /= weight_sum;
	rewardTarget /= (double)mu;

	std::cout << "current avg elite similarity reward: " << rewardTrajectory << ", target reward: " << rewardTarget << ", cutline: " << mPrevRewardTrajectory << std::endl;
	
	// if(mPrevRewardTrajectory < rewardTrajectory) {

		for(int i = 0; i < num_knot + 3; i++) {
		    mean_cps[i] /= weight_sum;
		    mPrevCps[i] = mPrevCps[i] * 0.6 + mean_cps[i] * 0.4;
		}

		mPrevRewardTrajectory = rewardTrajectory;
		mPrevRewardTarget = rewardTarget;

		mean_spline->SetControlPoints(0, mPrevCps);
		std::vector<Eigen::VectorXd> new_displacement = mean_spline->ConvertSplineToMotion();
		std::vector<Eigen::VectorXd> newpos;
		this->AddDisplacementToBVH(new_displacement, newpos);

		std::vector<Eigen::VectorXd> newvel = this->GetVelocityFromPositions(newpos);
		for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
			mMotions_phase_adaptive[i]->SetPosition(newpos[i]);
			mMotions_phase_adaptive[i]->SetVelocity(newvel[i]);
		}
	
		this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase_adaptive, mMotions_gen_adaptive);
		this->SaveAdaptiveMotion();
		this->SaveAdaptiveMotion(std::to_string(nOp));

		//save motion
		path =  mPath + std::string("motion") + std::to_string(nOp);
		ofs.open(path);

		for(auto t: newpos) {	
			ofs << t.transpose() << std::endl;
		}
		ofs.close();

		nOp += 1;
			
		while(!mSamples.empty()){
			MultilevelSpline* s = std::get<0>(mSamples.back());
			mSamples.pop_back();

			delete s;
		}	

		for(int i = 0; i < nRejectedSamples.size(); i++) {
			std::cout << i << " " << nRejectedSamples[i] << std::endl;
			nRejectedSamples[i] = 0;
		}

		Eigen::VectorXd meanTarget(mSampleTargets[0].rows());
		meanTarget.setZero();
		for(int i = 0; i < mSampleTargets.size(); i++) {
			meanTarget += mSampleTargets[i];
		}
		meanTarget /= mSampleTargets.size();

		if(meanTarget(0) + 5 >= mTargetRefUpdate) {
			std::cout << "target updated from " << mTargetRefUpdate  << " to " << meanTarget(0) + 5 << std::endl;
			mTargetRefUpdate = meanTarget(0) + 5;
		}
		mSampleTargets.clear();
		return true;
	// } else {
	// 	while(mSamples.size() > 100){
	// 		MultilevelSpline* s = mSamples.back().first;
	// 		mSamples.pop_back();

	// 		delete s;
	// 	}
	// }
	// return false;
}
std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> 
ReferenceManager::
GetRegressionSamples() {
	std::vector<Eigen::VectorXd> x;
	std::vector<Eigen::VectorXd> y;
	
	for(int i = 0; i < mRegressionSamples.size(); i++) {
		std::pair<std::vector<Eigen::VectorXd>, Eigen::VectorXd> s = mRegressionSamples[i];
		for(int j = 0; j < mKnots.size() + 3; j++) {
			Eigen::VectorXd knot_and_target;
			
			knot_and_target.resize(1 + s.second.rows());
			knot_and_target << j, s.second;
			x.push_back(knot_and_target);
			y.push_back((s.first)[j]);
		}
	}
	mRegressionSamples.clear();

	return std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>(x, y);
}

};