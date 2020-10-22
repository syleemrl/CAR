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
SaveAdaptiveMotion(std::string postfix) {
	std::string path = mPath + std::string("adaptive") + postfix;
	std::cout << "save motion to:" << path << std::endl;

	std::ofstream ofs(path);

	for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
		ofs << mMotions_phase_adaptive[i]->GetPosition().transpose() << std::endl;
		ofs << mMotions_phase_adaptive[i]->GetVelocity().transpose() << std::endl;
		ofs << mTimeStep_adaptive[i] << std::endl;

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
	ofs << mKnots_t.size() << std::endl;
	for(auto t: mKnots_t) {	
		ofs << t << std::endl;
	}
	for(auto t: mPrevCps_t) {	
		ofs << t.transpose() << std::endl;
	}
	ofs.close();

	path = mPath + std::string("time") + postfix;
	std::cout << "save results to" << path << std::endl;
	
	ofs.open(path);
	ofs << mPhaseLength << std::endl;

	for(int i = 0; i < mPhaseLength; i++) {
		ofs << i << " " << i << " " << mTimeStep_adaptive[i] << std::endl;
	}
	ofs.close();

}
void 
ReferenceManager::
LoadAdaptiveMotion(std::vector<Eigen::VectorXd> cps) {

	std::vector<Eigen::VectorXd> cps_space;
	std::vector<Eigen::VectorXd> cps_time;

	for(int i = 0 ; i < cps.size(); i++) {
		cps_space.push_back(cps[i].head(cps[i].rows()-1));
		cps_time.push_back(cps[i].tail(1));
	}
	std::vector<int> nc;
	nc.push_back(3);
	nc.push_back(5);
	DPhy::MultilevelSpline* s = new DPhy::MultilevelSpline(1, mPhaseLength, nc);

	s->SetKnots(0, mKnots);
	s->SetControlPoints(0, cps_space);

	DPhy::MultilevelSpline* st = new DPhy::MultilevelSpline(1, mPhaseLength);
	st->SetKnots(0, mKnots);
	st->SetControlPoints(0, cps_time);

	std::vector<Eigen::VectorXd> newpos;
	std::vector<Eigen::VectorXd> new_displacement = s->ConvertSplineToMotion();
	this->AddDisplacementToBVH(new_displacement, newpos);
	std::vector<Eigen::VectorXd> newvel = this->GetVelocityFromPositions(newpos);

	for(int j = 0; j < mPhaseLength; j++) {
		mMotions_phase_adaptive[j]->SetPosition(newpos[j]);
		mMotions_phase_adaptive[j]->SetVelocity(newvel[j]);
	}

	std::vector<Eigen::VectorXd> new_displacement_t = st->ConvertSplineToMotion();

	for(int i = 0; i < mPhaseLength; i++) {
		mTimeStep_adaptive[i] = 1 + new_displacement_t[i](0);
	}

	delete s;
	delete st;
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
		is >> buffer;
		mTimeStep_adaptive[i] = atof(buffer);
	}
	is.close();
	
	path = mPath + std::string("cp") + postfix;
	std::cout << "load Motion from: " << path << std::endl;

	is.open(path);
	mKnots.clear();

	is >> buffer;
	int knot_size = atoi(buffer);
	for(int i = 0; i < knot_size; i++) {	
		is >> buffer;
		mKnots.push_back(atoi(buffer));
	}
	for(int i = 0; i < knot_size + 3; i++) {	
		Eigen::VectorXd cps(mDOF);	
		for(int j = 0; j < mDOF; j++) {
			is >> buffer;
			cps[j] = atof(buffer);
		}
		mPrevCps[i] = cps;
	}

	mKnots_t.clear();

	is >> buffer;
	knot_size = atoi(buffer);

	for(int i = 0; i < knot_size; i++) {	
		is >> buffer;
		mKnots_t.push_back(atoi(buffer));
	}
	for(int i = 0; i < knot_size + 3; i++) {	
		Eigen::VectorXd cps_t(1);	
		is >> buffer;
		cps_t(0) = atof(buffer);
		mPrevCps_t[i] = cps_t;
	}

	is.close();

	this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase_adaptive, mMotions_gen_adaptive);

}
void 
ReferenceManager::
LoadMotionFromBVH(std::string filename)
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
	contact.push_back("RightToe");
	contact.push_back("RightFoot");
	contact.push_back("LeftToe");
	contact.push_back("LeftFoot");

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

	this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase, mMotions_gen);

	for(int i = 0; i < this->GetPhaseLength(); i++) {
		mMotions_phase_adaptive.push_back(new Motion(mMotions_phase[i]));
	}
	this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase_adaptive, mMotions_gen_adaptive);
}
std::vector<Eigen::VectorXd> 
ReferenceManager::
GetVelocityFromPositions(std::vector<Eigen::VectorXd> pos)
{
	std::vector<Eigen::VectorXd> vel;
	auto skel = mCharacter->GetSkeleton();
	for(int i = 0; i < pos.size() - 1; i++) {
		skel->setPositions(pos[i]);
		skel->computeForwardKinematics(true,false,false);
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
void 
ReferenceManager::
GenerateMotionsFromSinglePhase(int frames, bool blend, std::vector<Motion*>& p_phase, std::vector<Motion*>& p_gen)
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

	Eigen::Vector3d p0_footl = skel->getBodyNode("LeftFoot")->getWorldTransform().translation();
	Eigen::Vector3d p0_footr = skel->getBodyNode("RightFoot")->getWorldTransform().translation();


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

				Eigen::Vector3d p_footl = skel->getBodyNode("LeftFoot")->getWorldTransform().translation();
				Eigen::Vector3d p_footr = skel->getBodyNode("RightFoot")->getWorldTransform().translation();

				p_footl(1) = p0_footl(1);
				p_footr(1)= p0_footr(1);

				constraints.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>("LeftFoot", p_footl, Eigen::Vector3d(0, 0, 0)));
				constraints.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>("RightFoot", p_footr, Eigen::Vector3d(0, 0, 0)));

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
Eigen::VectorXd 
ReferenceManager::
GetPosition(double t , bool adaptive) 
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
Motion*
ReferenceManager::
GetMotion(double t, bool adaptive)
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
	else {
		return new Motion(DPhy::BlendPosition((*p_gen)[k1]->GetPosition(), (*p_gen)[k0]->GetPosition(), 1 - (t-k0)), 
				DPhy::BlendVelocity((*p_gen)[k1]->GetVelocity(), (*p_gen)[k0]->GetVelocity(), 1 - (t-k0)));		
	}
}
void
ReferenceManager::
ResetOptimizationParameters(bool reset_cps) {
	mExplorationMode = true;
	if(reset_cps) {
		mPrevCps.clear();
		for(int i = 0; i < this->mKnots.size() + 3; i++) {
			mPrevCps.push_back(Eigen::VectorXd::Zero(mDOF));
		}
		
		mPrevCps_t.clear();
		for(int i = 0; i < this->mKnots_t.size() + 3; i++) {
			mPrevCps_t.push_back(Eigen::VectorXd::Zero(1));
		}

		mTimeStep_adaptive.clear();
		for(int i = 0; i < mPhaseLength; i++) {
			mTimeStep_adaptive.push_back(1.0);
		}

		mMotions_phase_adaptive.clear();
		for(int i = 0; i < this->GetPhaseLength(); i++) {
			mMotions_phase_adaptive.push_back(new Motion(mMotions_phase[i]));
		}
		this->GenerateMotionsFromSinglePhase(1000, false, mMotions_phase_adaptive, mMotions_gen_adaptive);

	}

	nOp = 0;
	
	mPrevRewardTrajectory = 0.0;
	if(isParametric) {
		mRegressionMemory->ResetPrevSpace();
		mPrevRewardParam = mRegressionMemory->GetParamReward(mParamBVH, mParamGoal);	
	}
	else 
		mPrevRewardParam = 0;
	mMeanTrackingReward = 0;
	mMeanParamReward = 0;
	mPrevMeanParamReward = 0;

	mSamples.clear();
	mSampleParams.clear();

	nET = 0;
	nT = 0;
	nProgress = 0;
}
void
ReferenceManager::
InitOptimization(int nslaves, std::string save_path, bool parametric) {
	isParametric = parametric;
	mPath = save_path;
	

	mThresholdTracking = 0.91;
	mThresholdSurvival = 0.8;
	mThresholdProgress = 10;

	for(int i = 0; i <= 20; i+=4) {
		mKnots.push_back(i);
	} 
	mKnots.push_back(27);
	mKnots.push_back(38);
	mKnots.push_back(44);
	mKnots.push_back(49);
	mKnots.push_back(57);

	// for(int i = 0; i < mPhaseLength; i+= 4) {
	// 	mKnots_t.push_back(i);
	// }
	mKnots_t = mKnots;

	mParamBVH.resize(4);
	mParamBVH << 0.707107, 1.3, 1.2, 0.1;

	mParamCur.resize(4);
	mParamCur << 0.707107, 1.3, 1.2, 0.1;

	mParamGoal.resize(4);
	mParamGoal << 0.707107, 1.3, 1.2, 0.1;

	if(isParametric) {
		Eigen::VectorXd paramUnit(4);
		paramUnit<< 0.1, 0.1, 0.1, 0.1;

		mParamBase.resize(4);
		mParamBase << 0, 1.0, 0.8, 0.0;

		mParamEnd.resize(4);
		mParamEnd << 0.8, 1.5, 1.4, 0.6;
		// Eigen::VectorXd paramUnit(4);
		// paramUnit<< 0.1, 0.1, 0.1, 0.1;

		// mParamBase.resize(4);
		// mParamBase << 0.0, 1.1, -1.2, 0.0;

		// mParamEnd.resize(4);
		// mParamEnd << 1.0, 1.5, -0.8, 0.6;
		mRegressionMemory->InitParamSpace(mParamCur, std::pair<Eigen::VectorXd, Eigen::VectorXd> (mParamBase, mParamEnd), 
										  paramUnit, mDOF + 1, mKnots.size() + 3);
		mParamGoal = mRegressionMemory->SelectNewParamGoal();
	}

	ResetOptimizationParameters();

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
	contact.push_back("RightFoot");
	contact.push_back("RightToe");
	contact.push_back("LeftFoot");
	contact.push_back("LeftToe");

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
double 
ReferenceManager::
GetTimeStep(double t, bool adaptive) {
	if(adaptive) {
		t = std::fmod(t, mPhaseLength);
		int k0 = (int) std::floor(t);
		int k1 = (int) std::ceil(t);	
		if (k0 == k1) {
			return mTimeStep_adaptive[k0];
		}
		else if(k1 >= mTimeStep_adaptive.size())
			return (1 - (t - k0)) * mTimeStep_adaptive[k0] + (t-k0) * mTimeStep_adaptive[0];
		else
			return (1 - (t - k0)) * mTimeStep_adaptive[k0] + (t-k0) * mTimeStep_adaptive[k1];
	} else 
		return 1.0;
}
void
ReferenceManager::
ReportEarlyTermination() {
	mLock_ET.lock();
	nET +=1;
	mLock_ET.unlock();
}
void 
ReferenceManager::
SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, 
				 std::pair<double, double> rewards,
				 Eigen::VectorXd parameters) {
	if(dart::math::isNan(rewards.first) || dart::math::isNan(rewards.second)) {
		mLock_ET.lock();
		nET +=1;
		mLock_ET.unlock();
		return;
	}
	
	mLock_ET.lock();
	nT += 1;
	mLock_ET.unlock();
	mMeanTrackingReward = 0.99 * mMeanTrackingReward + 0.01 * rewards.first;
	mMeanParamReward = 0.99 * mMeanParamReward + 0.01 * rewards.second;
	std::vector<int> flag;

	if(rewards.first < mThresholdTracking) {
		flag.push_back(0);
	}
	else {
		flag.push_back(1);
	}

	if(rewards.second < mPrevRewardParam)
		flag.push_back(0);
	else
		flag.push_back(1);

	if(flag[0] == 0)
		return;

	std::vector<int> nc;
	nc.push_back(3);
	nc.push_back(5);

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

	nc.clear();
	nc.push_back(0);
	MultilevelSpline* st = new MultilevelSpline(1, this->GetPhaseLength());
	st->SetKnots(0, mKnots_t);

	std::vector<std::pair<Eigen::VectorXd,double>> displacement_t;
	for(int i = 0; i < data_spline.size(); i++) {
		double phase = std::fmod(data_spline[i].second, mPhaseLength);
		if(i < data_spline.size() - 1) {
			Eigen::VectorXd ts(1);
			ts << data_spline[i+1].second - data_spline[i].second - 1;
			displacement_t.push_back(std::pair<Eigen::VectorXd,double>(ts, phase));
		}
		else
			displacement_t.push_back(std::pair<Eigen::VectorXd,double>(Eigen::VectorXd::Zero(1), phase));
	}

	st->ConvertMotionToSpline(displacement_t);

	double r_slide = 0;
	std::vector<std::vector<std::pair<bool, Eigen::Vector3d>>> c;
	for(int i = 0; i < newpos.size(); i++) {
		c.push_back(this->GetContactInfo(newpos[i]));
	}
	for(int i = 1; i < newpos.size(); i++) {
		if(i < newpos.size() - 1) {
			for(int j = 0; j < c[i].size(); j++) {
				if((c[i-1][j].first) && (c[i+1][j].first) && !(c[i][j].first)) 
					(c[i][j].first) = true;
			}
		}
		std::vector<bool> flag_slide;
		double r_slide_frame = 0;
		for(int j = 0; j < c[i].size(); j++) {
			bool c_prev_j = c[i-1][j].first;
			bool c_cur_j = c[i][j].first;
			if(c_prev_j && c_cur_j) {
				double d = (c[i-1][j].second - c[i][j].second).norm(); 
				r_slide_frame += pow(d, 2);
				flag_slide.push_back(true);
			} 
			else
				flag_slide.push_back(false);
		}
		if((flag_slide[0] || flag_slide[1]) && ( flag_slide[2]|| flag_slide[3]))
			r_slide += 10 * r_slide_frame;
		else r_slide += r_slide_frame;
	}
	r_slide = exp(-r_slide);
	auto cps = s->GetControlPoints(0);
	double r_regul = 0;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < cps.size(); i++) {
		for(int j = 0; j < n_bnodes; j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			std::string b_name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
			if(dof == 6) {
				r_regul += 1 * cps[i].segment<3>(idx).norm();
				r_regul += 1 * cps[i].segment<3>(idx + 3).norm();
			} else if (dof == 3) {
				if(b_name.find("RightShoulder") != std::string::npos || 
				   b_name.find("RightArm") != std::string::npos ||
				   b_name.find("RightForeArm") != std::string::npos ||
				   b_name.find("RightHand") != std::string::npos) {
					r_regul += 2 * cps[i].segment<3>(idx).norm();
				} else
					r_regul += 0.5 * cps[i].segment<3>(idx).norm();
			} 
		}
	}
	r_regul = exp(-pow(r_regul / cps.size(), 2)*0.1);
	double reward_trajectory = (0.4 * r_regul + 0.6 * r_slide);;
	auto cps_t = st->GetControlPoints(0);

	mLock.lock();

	if(isParametric && reward_trajectory > 0.3) {
		auto cps_t = st->GetControlPoints(0);

		std::vector<Eigen::VectorXd> cps_tot;
		for(int i = 0; i < cps.size(); i++) {
			Eigen::VectorXd cps_temp(cps[0].rows() + 1);
			cps_temp << cps[i], cps_t[i];
			cps_tot.push_back(cps_temp);
		}
		mRegressionMemory->UpdateParamSpace(std::tuple<std::vector<Eigen::VectorXd>, Eigen::VectorXd, double>
											(cps_tot, parameters, reward_trajectory));
	}

	if(flag[1] && mExplorationMode) {
		mSamples.push_back(std::tuple<std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>, 
							std::pair<double, double>,  
							double>(std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>(s->GetControlPoints(0), st->GetControlPoints(0)), 
									std::pair<double, double>(reward_trajectory, r_slide), 
									rewards.second));
		mSampleParams.push_back(parameters);
		std::string path = mPath + std::string("samples") + std::to_string(nOp);

		std::ofstream ofs;
		ofs.open(path, std::fstream::out | std::fstream::app);

		for(auto t: data_spline) {	
			ofs << t.first.transpose() << " " << t.second << " " << r_slide << " " << r_regul << " " << rewards.second << std::endl;
		}
		ofs.close();
	} 	
	delete s;
	delete st;

	mLock.unlock();


}
bool cmp(const std::tuple<std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>, std::pair<double, double>, double> &p1, 
		 const std::tuple<std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>, std::pair<double, double>, double> &p2){
    if(std::get<1>(p1).first > std::get<1>(p2).first){
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
	if(!mExplorationMode)
		return false;

	double rewardParam = 0;
	double rewardTrajectory = 0;
    int mu = 60;
    std::cout << "num sample: " << mSamples.size() << std::endl;
    if(mSamples.size() < 500) {
    	return false;
    }

    std::stable_sort(mSamples.begin(), mSamples.end(), cmp);

	std::vector<int> nc;
	nc.push_back(3);
	nc.push_back(5);

	MultilevelSpline* mean_spline = new MultilevelSpline(1, this->GetPhaseLength()); 
	mean_spline->SetKnots(0, mKnots);

	nc.clear();
	nc.push_back(0);
	MultilevelSpline* mean_spline_t = new MultilevelSpline(1, this->GetPhaseLength()); 
	mean_spline_t->SetKnots(0, mKnots_t);

	std::vector<Eigen::VectorXd> mean_cps;   
   	mean_cps.clear();

   	std::vector<Eigen::VectorXd> mean_cps_t;
   	mean_cps_t.clear();

   	int num_knot = mean_spline->GetKnots(0).size();
   	for(int i = 0; i < num_knot + 3; i++) {
		mean_cps.push_back(Eigen::VectorXd::Zero(mDOF));
	}
   	
   	int num_knot_t = mean_spline_t->GetKnots(0).size();
	for(int i = 0;i < num_knot_t + 3; i++) {
		mean_cps_t.push_back(Eigen::VectorXd::Zero(1));
	}

	double weight_sum = 0;

	std::string path = mPath + std::string("rewards");
	std::ofstream ofs;
	ofs.open(path, std::fstream::out | std::fstream::app);

	for(int i = 0; i < mu; i++) {
		double w = log(mu + 1) - log(i + 1);
	    weight_sum += w;
	    std::vector<Eigen::VectorXd> cps = std::get<0>(mSamples[i]).first;
	    for(int j = 0; j < num_knot + 3; j++) {
			mean_cps[j] += w * cps[j];
	    }
	    std::vector<Eigen::VectorXd> cps_t = std::get<0>(mSamples[i]).second;
	    for(int j = 0; j < num_knot_t + 3; j++) {
			mean_cps_t[j] += w * cps_t[j];
	    }
	    rewardTrajectory += w * std::get<1>(mSamples[i]).first;
	    rewardParam += std::get<2>(mSamples[i]);
	    ofs << std::get<1>(mSamples[i]).second << " ";

	}
	ofs << std::endl;
	ofs.close();

	rewardTrajectory /= weight_sum;
	rewardParam /= (double)mu;

	std::cout << "current avg elite similarity reward: " << rewardTrajectory << ", Param reward: " << rewardParam << ", cutline: " << mPrevRewardTrajectory << std::endl;
	

	for(int i = 0; i < num_knot + 3; i++) {
	    mean_cps[i] /= weight_sum;
	    mPrevCps[i] = mPrevCps[i] * 0.6 + mean_cps[i] * 0.4;
	}
	for(int i = 0; i < num_knot_t + 3; i++) {
	    mean_cps_t[i] /= weight_sum;
		mPrevCps_t[i] = mPrevCps_t[i] * 0.6 + mean_cps_t[i] * 0.4;
	}

	mPrevRewardTrajectory = rewardTrajectory;
	mPrevRewardParam = rewardParam;

	mean_spline->SetControlPoints(0, mPrevCps);
	std::vector<Eigen::VectorXd> new_displacement = mean_spline->ConvertSplineToMotion();
	std::vector<Eigen::VectorXd> newpos;
	this->AddDisplacementToBVH(new_displacement, newpos);

	mean_spline_t->SetControlPoints(0, mPrevCps_t);
	std::vector<Eigen::VectorXd> new_displacement_t = mean_spline_t->ConvertSplineToMotion();

	for(int i = 0; i < mPhaseLength; i++) {
		mTimeStep_adaptive[i] = 1 + new_displacement_t[i](0);
	}

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
				

	mParamCur.setZero();
	for(int i = 0; i < mSampleParams.size(); i++) {
		mParamCur += mSampleParams[i];
	}
	mParamCur /= mSampleParams.size();
	std::cout << "currrent elite param mean: " << mParamCur.transpose() << std::endl;
 	
 	mSampleParams.clear();
 	mSamples.clear();
	delete mean_spline;
	delete mean_spline_t;
	
	return true;
}
bool
ReferenceManager::
UpdateParamManually() {

	double survival_ratio = (double)nT / (nET + nT);
	std::cout << "current mean tracking reward :" << mMeanTrackingReward  << ", survival ratio: " << survival_ratio << std::endl;

	if(survival_ratio > mThresholdSurvival && mMeanTrackingReward > mThresholdTracking - 0.05) {
		return true;
	}
	return false;
}
bool 
ReferenceManager::
CheckExplorationProgress() {
	if(nET + nT == 0)
		return true;
	double survival_ratio = (double)nT / (nET + nT);
	nT = 0;
	nET = 0;
	std::cout << survival_ratio << " " << (mMeanParamReward - mPrevMeanParamReward) <<std::endl;
	if(survival_ratio > mThresholdSurvival && (mMeanParamReward - mPrevMeanParamReward) < 2 * 1e-2) {
		nProgress += 1;
	} else {
		nProgress = 0;
	}
	mPrevMeanParamReward = mMeanParamReward;
	if((nProgress >= mThresholdProgress && mRegressionMemory->GetTimeFromLastUpdate() > mThresholdProgress) || 
	   (mRegressionMemory->GetTimeFromLastUpdate() > 2 * mThresholdProgress)) {
		return false;
	}
	return true;
}

};