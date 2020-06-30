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

	mInterestedBodies.clear();
	mInterestedBodies.push_back("Torso");
	mInterestedBodies.push_back("Spine");
	mInterestedBodies.push_back("Neck");
	mInterestedBodies.push_back("Head");

	mInterestedBodies.push_back("ArmL");
	mInterestedBodies.push_back("ForeArmL");
	mInterestedBodies.push_back("HandL");

	mInterestedBodies.push_back("ArmR");
	mInterestedBodies.push_back("ForeArmR");
	mInterestedBodies.push_back("HandR");

	mInterestedBodies.push_back("FemurL");
	mInterestedBodies.push_back("TibiaL");
	mInterestedBodies.push_back("FootL");
	mInterestedBodies.push_back("FootEndL");

	mInterestedBodies.push_back("FemurR");
	mInterestedBodies.push_back("TibiaR");
	mInterestedBodies.push_back("FootR");
	mInterestedBodies.push_back("FootEndR");
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
SaveAdaptiveMotion() {
	std::string path = mPath + std::string("adaptive");
	std::cout << "save motion to:" << path << std::endl;

	std::ofstream ofs(path);

	for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
		ofs << mMotions_phase_adaptive[i]->GetPosition().transpose() << std::endl;
		ofs << mMotions_phase_adaptive[i]->GetVelocity().transpose() << std::endl;
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
		for(int j = 0; j < mMotions_phase[0]->GetPosition().rows(); j++) 
		{
			is >> buffer;
			pos[j] = atof(buffer);
		}
		for(int j = 0; j < mMotions_phase[0]->GetVelocity().rows(); j++) 
		{
			is >> buffer;
			vel[j] = atof(buffer);
		}
		mMotions_phase_adaptive[i]->SetPosition(pos);
		mMotions_phase_adaptive[i]->SetVelocity(vel);
	}
	is.close();

	this->GenerateMotionsFromSinglePhase(1000, false, true);
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
	int dof = skel->getPositions().rows();
	mDOF = dof;
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

		t += bvh->GetTimeStep();
	}
	mMotions_raw.back()->SetVelocity(mMotions_raw.front()->GetVelocity());

	mPhaseLength = mMotions_raw.size();
	mTimeStep = bvh->GetTimeStep();

	for(int i = 0; i < mPhaseLength; i++) {
		mMotions_phase.push_back(new Motion(mMotions_raw[i]));
	}

	delete bvh;

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
void
ReferenceManager::
InitOptimization(std::string save_path) {
	mKnots.push_back(0);
	mKnots.push_back(12);
	mKnots.push_back(29);
	mKnots.push_back(37);
	mKnots.push_back(44);
	mKnots.push_back(56);
	mKnots.push_back(64);
	mKnots.push_back(76);

	for(int i = 0; i < this->GetPhaseLength(); i++) {
		mMotions_phase_adaptive.push_back(new Motion(mMotions_phase[i]));
	}
	this->GenerateMotionsFromSinglePhase(1000, false, true);

	mPath = save_path;
}
void 
ReferenceManager::
SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, double rewards) {

	MultilevelSpline* s = new MultilevelSpline(1, this->GetPhaseLength());
	s->SetKnots(0, mKnots);

	std::vector<std::pair<Eigen::VectorXd,double>> displacement;
	for(int i = 0; i < data_spline.size(); i++) {

		Eigen::VectorXd p = data_spline[i].first;
		Eigen::VectorXd p_bvh = this->GetPosition(data_spline[i].second);
		Eigen::VectorXd d(mCharacter->GetSkeleton()->getNumDofs() + 1);
		int count = 0;
		for(int j = 0; j < mInterestedBodies.size(); j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[j])->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[j])->getParentJoint()->getNumDofs();
			if(dof == 6) {
				d.segment<3>(count) = JointPositionDifferences(p.segment<3>(count), p_bvh.segment<3>(count));
				d.segment<3>(count + 3) = p.segment<3>(count + 3) -  p_bvh.segment<3>(count + 3);
			} else if (dof == 3) {
				d.segment<3>(count) = JointPositionDifferences(p.segment<3>(count), p_bvh.segment<3>(count));
			} else {
				d(count) = p(count) - p_bvh(count);
			}
			count += dof;
		}
		d.tail<1>() = p.tail<1>();
		displacement.push_back(std::pair<Eigen::VectorXd,double>(d, data_spline[i].second));
	}
	s->ConvertMotionToSpline(displacement);

	mLock.lock();
	mSamples.push_back(std::pair<MultilevelSpline*, double>(s, rewards));
	mLock.unlock();

	// std::string path = std::string(CAR_DIR)+std::string("/result/trajectory")+std::to_string(mSamples.size());
	// s->Save(path);
	
	// std::ofstream ofs(path, std::fstream::out | std::fstream::app);
	// ofs << data_spline.size() << std::endl;
	// for(auto t: data_spline) {
	// 	ofs << t.second << std::endl;
	// 	ofs << t.first.transpose() << std::endl;
	// }
	// std::cout << "saved trajectory to " << path << std::endl;
	// ofs.close();
}
bool cmp(const std::pair<DPhy::MultilevelSpline*, double> &p1, const std::pair<DPhy::MultilevelSpline*, double> &p2){
    if(p1.second > p2.second){
        return true;
    }
    else{
        return false;
    }
}
void 
ReferenceManager::
Optimize() {
    std::stable_sort(mSamples.begin(), mSamples.end(), cmp);
    int mu = std::floor(mSamples.size() / 2.0);
	MultilevelSpline* mean_spline = new MultilevelSpline(1, this->GetPhaseLength()); 
	mean_spline->SetKnots(0, mKnots);

	std::vector<Eigen::VectorXd> mean_cps;   
   	mean_cps.clear();
   	int num_knot = mean_spline->GetKnots(0).size();
   	for(int i = 0; i < num_knot; i++) {
		mean_cps.push_back(Eigen::VectorXd::Zero(mDOF));
	}
	double weight_sum = 0;

	for(int i = 0; i < mu; i++) {
		double w = log(mu + 1) - log(i + 1);
	    weight_sum += w;
	    std::vector<Eigen::VectorXd> cps = mSamples[i].first->GetControlPoints(0);
	    for(int j = 0; j < num_knot; j++) {
			mean_cps[j] += w * cps[j].head(cps[j].rows() - 1);
	    }
	}
	for(int i = 0; i < num_knot; i++) {
	    mean_cps[i] /= weight_sum;
	}
	mean_spline->SetControlPoints(0, mean_cps);
   	std::vector<Eigen::VectorXd> new_displacement = mean_spline->ConvertSplineToMotion();
	std::vector<Eigen::VectorXd> newpos;
	
	for(int i = 0; i < new_displacement.size(); i++) {

		Eigen::VectorXd p_bvh = mMotions_phase[i]->GetPosition();
		Eigen::VectorXd d = new_displacement[i];
		Eigen::VectorXd p(mCharacter->GetSkeleton()->getNumDofs());

		int count = 0;
		for(int j = 0; j < mInterestedBodies.size(); j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[j])->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[j])->getParentJoint()->getNumDofs();
			if(dof == 6) {
				p.segment<3>(count) = Rotate3dVector(p_bvh.segment<3>(count), d.segment<3>(count));
				p.segment<3>(count + 3) = d.segment<3>(count + 3) + p_bvh.segment<3>(count + 3);
			} else if (dof == 3) {
				p.segment<3>(count) = Rotate3dVector(p_bvh.segment<3>(count), d.segment<3>(count));
			} else {
				p(count) = d(count) + p_bvh(count);
			}
			count += dof;
		}
		newpos.push_back(p);
	}
	std::vector<Eigen::VectorXd> newvel = this->GetVelocityFromPositions(newpos);
	for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
		mMotions_phase_adaptive[i]->SetPosition(newpos[i]);
		mMotions_phase_adaptive[i]->SetVelocity(newvel[i]);
	}
	this->GenerateMotionsFromSinglePhase(1000, false, true);


	std::string path = std::string(CAR_DIR) + std::string("/result/op_temp");

	std::ofstream ofs(path);

	for(auto t: newpos) {	
		ofs << t.transpose() << std::endl;
	}
	ofs.close();

	while(!mSamples.empty()){
		MultilevelSpline* s = mSamples.back().first;
		mSamples.pop_back();

		delete s;

	}
}
};