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
	this->ComputeAxisMean();
	this->ComputeAxisDev();

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
InitOptimization(std::string save_path) {
	mKnots.push_back(0);
	mKnots.push_back(12);
	mKnots.push_back(29);
	mKnots.push_back(37);
	mKnots.push_back(44);
	mKnots.push_back(56);
	mKnots.push_back(64);
	mKnots.push_back(76);

	for(int i = 0; i < this->mKnots.size(); i++) {
		mPrevCps.push_back(Eigen::VectorXd::Zero(mDOF));
	}
	for(int i = 0; i < this->GetPhaseLength(); i++) {
		mMotions_phase_adaptive.push_back(new Motion(mMotions_phase[i]));
	}
	this->GenerateMotionsFromSinglePhase(1000, false, true);
	
	nOp = 0;
	mPath = save_path;
	mPrevRewardTrajectory = 0;
}
void 
ReferenceManager::
SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, std::pair<double, double> rewards) {

	if((rewards.first / mPhaseLength)  < 0.8)
		return;

	MultilevelSpline* s = new MultilevelSpline(1, this->GetPhaseLength());
	s->SetKnots(0, mKnots);

	std::vector<std::pair<Eigen::VectorXd,double>> displacement;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < data_spline.size(); i++) {

		Eigen::VectorXd p = data_spline[i].first;
		Eigen::VectorXd p_bvh = this->GetPosition(data_spline[i].second);
		Eigen::VectorXd d(mCharacter->GetSkeleton()->getNumDofs() + 1);
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
		d.tail<1>() = p.tail<1>();
		displacement.push_back(std::pair<Eigen::VectorXd,double>(d, std::fmod(data_spline[i].second, mPhaseLength)));
	}
	s->ConvertMotionToSpline(displacement);

	auto cps = s->GetControlPoints(0);
	double r = 0;
	for(int i = 0; i < cps.size(); i++) {
		r += cps[i].norm();	
	}
//	std::cout << rewards << " " << exp(-pow(r, 2)*0.01) << " "<< 0.05*exp(-pow(r, 2)*0.01) + rewards<< std::endl;
//	std::cout << rewards << std::endl;
//	std::cout << rewards.first / mPhaseLength << " " << rewards.second << std::endl;

	double reward_trajectory = rewards.second; // + 0.2 * exp(-pow(r, 2)*0.005);
	mLock.lock();
	mSamples.push_back(std::pair<MultilevelSpline*, double>(s, reward_trajectory));
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

	double rewardTrajectory = 0;
    int mu = 100;
    std::cout << "num sample: " << mSamples.size() << std::endl;
    if(mSamples.size() < 200)
    	return;

    std::stable_sort(mSamples.begin(), mSamples.end(), cmp);
	MultilevelSpline* mean_spline = new MultilevelSpline(1, this->GetPhaseLength()); 
	mean_spline->SetKnots(0, mKnots);

	std::vector<Eigen::VectorXd> mean_cps;   
   	mean_cps.clear();
   	int num_knot = mean_spline->GetKnots(0).size();
   	for(int i = 0; i < num_knot; i++) {
		mean_cps.push_back(Eigen::VectorXd::Zero(mDOF));
	}
	double weight_sum = 0;

	std::string path = mPath + std::string("rewards");
	std::ofstream ofs;
	ofs.open(path, std::fstream::out | std::fstream::app);

	for(int i = 0; i < mu; i++) {
		double w = log(mu + 1) - log(i + 1);
	    weight_sum += w;
	    std::vector<Eigen::VectorXd> cps = mSamples[i].first->GetControlPoints(0);
	    for(int j = 0; j < num_knot; j++) {
			mean_cps[j] += w * cps[j].head(cps[j].rows() - 1);
	    }
	    rewardTrajectory += w * mSamples[i].second;
	    ofs << mSamples[i].second << " ";
	}
	ofs << std::endl;
	ofs.close();
	for(int i = 0; i < num_knot; i++) {
	    mean_cps[i] /= weight_sum;
	    mPrevCps[i] = mPrevCps[i] * 0.95 + mean_cps[i] * 0.05;
	}
	rewardTrajectory /= weight_sum;
	
	std::cout << "prev avg elite reward: " << mPrevRewardTrajectory << " current avg elite reward: " << rewardTrajectory << std::endl;

	if(rewardTrajectory > mPrevRewardTrajectory) {
		mean_spline->SetControlPoints(0, mPrevCps);
	   	std::vector<Eigen::VectorXd> new_displacement = mean_spline->ConvertSplineToMotion();
		std::vector<Eigen::VectorXd> newpos;
		int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

		for(int i = 0; i < new_displacement.size(); i++) {

			Eigen::VectorXd p_bvh = mMotions_phase[i]->GetPosition();
			Eigen::VectorXd d = new_displacement[i];
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
			newpos.push_back(p);
		}
		std::vector<Eigen::VectorXd> newvel = this->GetVelocityFromPositions(newpos);
		for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
			mMotions_phase_adaptive[i]->SetPosition(newpos[i]);
			mMotions_phase_adaptive[i]->SetVelocity(newvel[i]);
		}
		this->GenerateMotionsFromSinglePhase(1000, false, true);
		this->SaveAdaptiveMotion();
		this->SaveAdaptiveMotion(std::to_string(nOp));

		//save control points
		path = mPath + std::string("cp") + std::to_string(nOp);
		ofs.open(path);
		ofs << mKnots.size() << std::endl;
		for(auto t: mKnots) {	
			ofs << t << std::endl;
		}
		for(auto t: mean_cps) {	
			ofs << t.transpose() << std::endl;
		}
		ofs.close();

		//save motion
		path =  mPath + std::string("motion") + std::to_string(nOp);
		ofs.open(path);

		for(auto t: newpos) {	
			ofs << t.transpose() << std::endl;
		}
		ofs.close();


		nOp += 1;
		mPrevRewardTrajectory = rewardTrajectory;
		
		while(!mSamples.empty()){
			MultilevelSpline* s = mSamples.back().first;
			mSamples.pop_back();

			delete s;
		}	
	}
}
};