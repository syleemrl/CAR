#include "ReferenceManager.h"
#include <tinyxml.h>
#include <fstream>
#include <stdlib.h>
#include <cmath>

using namespace dart::dynamics;
namespace DPhy
{
ReferenceManager::ReferenceManager(Character* character) 
:mRD(), mMT(mRD()), mUniform(0.0, 1.0)
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
LoadAdaptiveMotion(std::vector<Eigen::VectorXd> displacement) {

	std::vector<Eigen::VectorXd> d_space;
	std::vector<Eigen::VectorXd> d_time;

	for(int i = 0 ; i < displacement.size(); i++) {
		d_space.push_back(displacement[i].head(displacement[i].rows()-1));
		d_time.push_back(displacement[i].tail(1));
	}

	std::vector<Eigen::VectorXd> newpos;
	this->AddDisplacementToBVH(d_space, newpos);
	std::vector<Eigen::VectorXd> newvel = this->GetVelocityFromPositions(newpos);

	for(int j = 0; j < mPhaseLength; j++) {
		mMotions_phase_adaptive[j]->SetPosition(newpos[j]);
		mMotions_phase_adaptive[j]->SetVelocity(newvel[j]);
	}


	for(int i = 0; i < mPhaseLength; i++) {
		mTimeStep_adaptive[i] = exp(d_time[i](0));
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
		is >> buffer;
		mTimeStep_adaptive[i] = atof(buffer);
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
	if(reset_cps) {
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
	
	if(isParametric) {
		mRegressionMemory->ResetExploration();
	}

	mMeanTrackingReward = 0;
	mMeanParamReward = 0;
	mPrevMeanParamReward = 0;

	nET = 0;
	nT = 0;
	nProgress = 0;
}
void
ReferenceManager::
InitOptimization(int nslaves, std::string save_path, bool adaptive) {
	isParametric = adaptive;
	mPath = save_path;
	

	mThresholdTracking = 0.87;
	mThresholdSurvival = 0.8;
	mThresholdProgress = 10;

		mParamBVH.resize(4);
	mParamBVH << 0.707107, 1.3, 1.2, 0.36;

	mParamCur.resize(4);
	mParamCur << 0.707107, 1.3, 1.2, 0.36;

	mParamGoal.resize(4);
	mParamGoal << 0.707107, 1.3, 1.2, 0.36;

	if(isParametric) {
		Eigen::VectorXd paramUnit(4);
		paramUnit<< 0.1, 0.1, 0.1, 0.1;

		mParamBase.resize(4);
		mParamBase << 0.2, 1.1, 0.8, 0.1;

		mParamEnd.resize(4);
		mParamEnd << 0.8, 1.4, 1.2, 0.6;

		
		mRegressionMemory->InitParamSpace(mParamCur, std::pair<Eigen::VectorXd, Eigen::VectorXd> (mParamBase, mParamEnd), 
										  paramUnit, mDOF + 1, mPhaseLength);


		std::cout << "initial goal : " << mParamGoal.transpose() << std::endl;
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
				 std::tuple<double, double, double> rewards,
				 Eigen::VectorXd parameters) {
	if(dart::math::isNan(std::get<0>(rewards)) || dart::math::isNan(std::get<1>(rewards))) {
		mLock_ET.lock();
		nET +=1;
		mLock_ET.unlock();
		return;
	}
	mLock_ET.lock();
	nT += 1;
	mLock_ET.unlock();
	mMeanTrackingReward = 0.99 * mMeanTrackingReward + 0.01 * std::get<0>(rewards);
	mMeanParamReward = 0.99 * mMeanParamReward + 0.01 * std::get<1>(rewards);
	std::vector<int> flag;
	if(std::get<0>(rewards) < mThresholdTracking) {
		flag.push_back(0);
	}
	else {
		flag.push_back(1);
	}
	if(flag[0] == 0)
		return;
	double start_phase = std::fmod(data_spline[0].second, mPhaseLength);
	std::vector<Eigen::VectorXd> trajectory;
	for(int i = 0; i < data_spline.size(); i++) {
		trajectory.push_back(data_spline[i].first);
	}
	trajectory = Align(trajectory, this->GetPosition(start_phase).segment<6>(0));
	for(int i = 0; i < data_spline.size(); i++) {
		data_spline[i].first = trajectory[i];
	}

	std::vector<std::pair<Eigen::VectorXd,double>> data_uniform;
	int count = 0;
	for(int i = 0; i < mPhaseLength; i++) {
		while(count + 1 < data_spline.size() && i >= data_spline[count+1].second)
			count += 1;
		Eigen::VectorXd p(mDOF + 1);
		if(i < data_spline[count].second) {
			int size = data_spline.size();
			double t0 = data_spline[size-1].second - data_spline[size-2].second;
			double weight = 1.0 - (mPhaseLength + i - data_spline[size-1].second) / (mPhaseLength + data_spline[count].second - data_spline[size-1].second);
			double t1 = data_spline[count+1].second - data_spline[count].second;
			Eigen::VectorXd p_blend = DPhy::BlendPosition(data_spline[size-1].first, data_spline[0].first, weight);
			double t_blend = (1 - weight) * t0 + weight * t1;
			p << p_blend, log(t_blend);
		} else if(count == data_spline.size() - 1 && i > data_spline[count].second) {
			double t0 = data_spline[count].second - data_spline[count-1].second;
			double weight = 1.0 - (data_spline[0].second + mPhaseLength - i) / (data_spline[0].second + mPhaseLength - data_spline[count].second);
			double t1 = data_spline[1].second - data_spline[0].second;
			
			Eigen::VectorXd p_blend = DPhy::BlendPosition(data_spline[count].first, data_spline[0].first, weight);
			double t_blend = (1 - weight) * t0 + weight * t1;
			p << p_blend, log(t_blend);
		} else if(i == data_spline[count].second) {
			if(count < data_spline.size())
				p << data_spline[count].first, log(data_spline[count+1].second - data_spline[count].second);
			else
				p << data_spline[count].first, log(data_spline[0].second + mPhaseLength - data_spline[count].second);

		} else {
			double weight = 1.0 - (data_spline[count+1].second - i) / (data_spline[count+1].second - data_spline[count].second);
			Eigen::VectorXd p_blend = DPhy::BlendPosition(data_spline[count].first, data_spline[count+1].first, weight);
			double t_blend;
			if(count + 2 >= data_spline.size()) {
				double t0 = data_spline[count+1].second - data_spline[count].second;
				double t1 = data_spline[1].second - data_spline[0].second;
				t_blend = (1 - weight) * t0 + weight * t1;
			} else {
				double t0 = data_spline[count+1].second - data_spline[count].second;
				double t1 = data_spline[count+2].second - data_spline[count+1].second;
				t_blend = (1 - weight) * t0 + weight * t1;
			}
			p << p_blend, log(t_blend);
		}
		data_uniform.push_back(std::pair<Eigen::VectorXd,double>(p, i));
	}

	std::vector<std::pair<Eigen::VectorXd,double>> displacement;
	this->GetDisplacementWithBVH(data_uniform, displacement);

	std::vector<Eigen::VectorXd> d;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	// double r_regul = 0;
	// for(int i = 0; i < mPhaseLength; i++) {
	// 	Eigen::VectorXd d_t(mDOF + 1);
	// 	d_t << displacement[i].first, data_uniform[i].first.tail<1>();
	// 	d.push_back(d_t);
	// 	for(int j = 0; j < n_bnodes; j++) {
	// 		int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
	// 		int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
	// 		std::string b_name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
	// 		if(dof == 6) {
	// 			r_regul += 1 * displacement[i].first.segment<3>(idx).norm();
	// 			r_regul += 5 * displacement[i].first.segment<3>(idx + 3).norm();
	// 		} else if (dof == 3) {
	// 			r_regul += 0.25 * displacement[i].first.segment<3>(idx).norm();
	// 		}
	// 	}
	// }

	// r_regul = exp(-pow(r_regul / mPhaseLength, 2)*0.1);
	double reward_trajectory = std::get<2>(rewards);
	mLock.lock();

	if(isParametric) {
		mRegressionMemory->UpdateParamSpace(std::tuple<std::vector<Eigen::VectorXd>, Eigen::VectorXd, double>
											(d, parameters, reward_trajectory));

		// std::string path = mPath + std::string("samples") + std::to_string(nOp);

		// std::ofstream ofs;
		// ofs.open(path, std::fstream::out | std::fstream::app);
		// for(auto t: data_spline) {	
		// 	ofs << t.transpose() << std::endl;
		// }
		// for(auto t: d) {	
		// 	ofs << t.transpose() << " " << r_regul << std::endl;
		// }
		// ofs.close();
	}
	
	mLock.unlock();


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
void
ReferenceManager::
OptimizeExReference(){
	mCPS_exp = mRegressionMemory->GetCPSFromNearestParams(mParamGoal);
}
void 
ReferenceManager::
SelectReference(){
	// double r = mRegressionMemory->GetTrainedRatio();
	// if(r < 0.1) {
	// 	LoadAdaptiveMotion(mCPS_exp);
	// } else {
	// 	r = std::min(r * 1.5, 0.8);
	// 	if(mUniform(mMT) < r) {
	// 		LoadAdaptiveMotion(mCPS_reg);
	// 	} else {
			LoadAdaptiveMotion(mCPS_exp);
	// 	}
	// }
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
	   (mRegressionMemory->GetTimeFromLastUpdate() > mThresholdProgress)) {
		return false;
	}
	return true;
}

};