#include "ReferenceManager.h"
#include <tinyxml.h>
#include <fstream>
#include <stdlib.h>
#include <cmath>

#define TARGET_FRAME 44
using namespace dart::dynamics;
double cps_d[5][45] = {
{-0.00822713, 0.071199, -0.0175336, 0.00161151, 9.19429e-05, -8.3496e-05, -0.00156934, 0.0154037, -0.0115516, -1.41728e-05, 9.91624e-08, -1.20464e-05, -1.28838e-09, 1.67377e-08, 5.38684e-09, -7.40066e-08, 7.48063e-08, -6.26254e-09, -1.84317e-07, -5.54496e-07, 4.28175e-07, 2.54515e-07, -0.0140755, 0.0335634, -0.0440647, 0.270618, 0.376215, 0.0303925, -0.0293694, -0.0144004, -0.00509865, 0.0122003, 0.0107491, -0.00108846, -0.0298101, 0.0173578, 2.33874e-07, 0.0379116, -0.00961917, 0.0299699, -0.0338688, 0.00634255, -0.0319149, 0.0110293, 0}, 
{0.0150812, -0.130517, 0.0321413, -0.00325265, -0.000251988, -0.0013004, 0.00151223, -0.0158857, 0.0119334, -7.21215e-06, -5.77891e-07, -1.20837e-06, -2.46043e-10, -6.32853e-08, 1.02872e-09, 1.15941e-07, -1.32701e-07, 6.21337e-09, -3.9502e-09, 4.68344e-07, -5.6311e-07, -3.57286e-07, 0.0154261, -0.0350462, 0.0451702, -0.279204, -0.388759, -0.0239681, 0.0359792, 0.0379281, 0.0156548, -0.0443003, -0.044451, 0.019868, 0.0323933, -0.0326844, 2.33874e-07, -0.0695121, 0.0264649, -0.0658805, 0.0393043, -0.00218523, 0.0365085, -0.0348172, 0}, 
{0.0290964, -0.25181, 0.0620114, -0.0070344, -0.000102275, 0.000677583, 0.00472934, -0.0441604, 0.0330695, -3.05328e-05, -1.0096e-06, -2.75945e-05, 3.91307e-10, 1.89034e-07, -1.63609e-09, -2.02665e-07, -6.8548e-07, -8.22426e-09, 3.87191e-08, 7.85112e-07, 8.94751e-07, -9.17767e-07, 0.0383064, -0.0952139, 0.126877, -0.775493, -1.07632, -0.101274, 0.0679969, 0.0337402, 0.0125258, -0.0434652, -0.0523723, -0.000219449, 0.0842867, -0.0557558, 2.33874e-07, -0.157492, 0.0194953, -0.0934755, 0.118624, -0.0219227, 0.091152, -0.0465642, 0}, 
{0.0333077, -0.288251, 0.0709855, -0.0064434, 8.05948e-05, 0.00510804, 0.00764875, -0.0650598, 0.048581, -2.10825e-05, -1.27261e-06, -1.6798e-05, -5.49485e-10, -6.61424e-08, 2.29744e-09, 4.60018e-07, -1.0092e-06, 1.24779e-08, -2.74961e-06, 1.47102e-06, 2.35938e-06, -1.3499e-06, 0.050449, -0.1373, 0.188507, -1.14131, -1.5788, -0.188371, 0.0495336, 0.0212192, -0.0190262, 0.06499, 0.0790508, -0.0816419, 0.122395, -0.0712024, 2.33874e-07, -0.113176, 0.00225962, -0.0532849, 0.197003, -0.053198, 0.128492, -0.00850061, 0}, 
{0.0256123, -0.221654, 0.0545851, -0.00537735, 4.88567e-05, 0.00201118, 0.00379533, -0.0377025, 0.0282813, -2.76492e-05, -7.64256e-07, -2.42138e-05, 8.45377e-10, 1.64965e-08, -3.53459e-09, 2.78125e-08, -4.84145e-07, 6.85498e-09, 5.85376e-07, 6.95206e-07, -6.8474e-07, -7.23473e-07, 0.0348076, -0.0823168, 0.107739, -0.66241, -0.92109, -0.071467, 0.0740972, 0.0318809, 0.00512396, -0.0133654, -0.012633, -0.0168577, 0.0723956, -0.053572, 2.33874e-07, -0.121535, 0.0167493, -0.0783748, 0.136667, -0.038083, 0.0781359, -0.0238701, 0}
};
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

	while(!p_gen.empty()){
		Motion* m = p_gen.back();
		p_gen.pop_back();

		delete m;
	}		

	auto& skel = mCharacter->GetSkeleton();

	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform(p_phase[0]->GetPosition().head<6>());
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform(p_phase.back()->GetPosition().head<6>());

	Eigen::Isometry3d T0_gen = T0_phase;
	
	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();

	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());			
	T01.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));
	T01.translation()[1] = 0;

	for(int i = 0; i < frames; i++) {
		
		int phase = i % mPhaseLength;
		
		if(i < mPhaseLength) {
			p_gen.push_back(new Motion(p_phase[i]));
		} else {
			Eigen::VectorXd pos = p_phase[phase]->GetPosition();
			Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
			T_current = T0_phase.inverse()*T_current;
			T_current = T0_gen*T_current;

			pos.head<6>() = dart::dynamics::FreeJoint::convertToPositions(T_current);
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
Motion* ReferenceManager::GetMotionForOptimization(double t, int id) {
	auto& skel = mCharacter->GetSkeleton();

	if(mMotions_gen_temp[id].size()-1 < t) {
	 	return new Motion(mMotions_gen_temp[id].back()->GetPosition(), mMotions_gen_temp[id].back()->GetVelocity());
	}
	
	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	

	if (k0 == k1)
		return new Motion(mMotions_gen_temp[id][k0]);
	else
		return new Motion(DPhy::BlendPosition(mMotions_gen_temp[id][k1]->GetPosition(), mMotions_gen_temp[id][k0]->GetPosition(), 1 - (t-k0)), 
				DPhy::BlendPosition(mMotions_gen_temp[id][k1]->GetVelocity(), mMotions_gen_temp[id][k0]->GetVelocity(), 1 - (t-k0)));		
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
	mKnots.push_back(9);
	mKnots.push_back(20);
	mKnots.push_back(27);
	mKnots.push_back(35);
	
	for(int i = 0; i < this->mKnots.size(); i++) {
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
	mPrevRewardTrajectory = 0.5;
	mPrevRewardTarget = 0.5;	
	mOpMode = false;

	// std::vector<std::pair<Eigen::VectorXd,double>> pos;
	// for(int i = 0; i < mPhaseLength; i++) {
	// 	pos.push_back(std::pair<Eigen::VectorXd,double>(mAxis_BVH[i], i));
	// }
	// MultilevelSpline* s = new MultilevelSpline(1, mPhaseLength);
	// s->SetKnots(0, mKnots);

	// s->ConvertMotionToSpline(pos);
	// std::string path = std::string(CAR_DIR) + std::string("/result/op_axis_cmp");
	
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
		if(p[1] < 0.04) {
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
SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_spline, std::pair<double, double> rewards) {
	if((rewards.first / mPhaseLength)  < 0.9 || rewards.second < mPrevRewardTarget)
		return;

	MultilevelSpline* s = new MultilevelSpline(1, this->GetPhaseLength());
	s->SetKnots(0, mKnots);

	std::vector<Eigen::VectorXd> trajectory;
	for(int i = 0; i < data_spline.size(); i++) {
		trajectory.push_back(data_spline[i].first);
	}
	// trajectory = Align(trajectory, this->GetPosition(0).segment<6>(0));

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
				r_slide += exp(-pow(d, 2)*1000);
			} else {
				r_slide += 1;
			}
		}
	}
	r_slide /= (newpos.size() * 2);
	auto cps = s->GetControlPoints(0);
	double r_regul = 0;
	for(int i = 0; i < cps.size(); i++) {
		r_regul += cps[i].norm();	
	}

	double reward_trajectory = 0.2 * exp(-pow(r_regul, 2)*0.01) + 0.8 * (r_slide - 0.5);
	// if(reward_trajectory < mPrevRewardTrajectory)
	// 	return;

	mLock.lock();
	mSamples.push_back(std::tuple<MultilevelSpline*, double,  double>(s, reward_trajectory, rewards.second));
	// // if(nOp != 0) {
	// 	std::string path =  mPath + std::string("samples") + std::to_string(nOp);
		
	// 	std::ofstream ofs;

	// 	ofs.open(path, std::fstream::out | std::fstream::app);
	// 	for(auto t: data_spline) {	
	// 		ofs << t.first.transpose() << " " << rewards.first << std::endl;
	// 	}
	// 	ofs.close();
//	}

	mLock.unlock();

}
bool cmp(const std::tuple<DPhy::MultilevelSpline*, double, double> &p1, const std::tuple<DPhy::MultilevelSpline*, double, double> &p2){
    if(std::get<1>(p1) > std::get<1>(p2)){
        return true;
    }
    else{
        return false;
    }
}
void
ReferenceManager::
GenerateRandomTrajectory(int i) {
	
	std::vector<int> idxs;

	idxs.push_back(25);
	idxs.push_back(26);
	idxs.push_back(27);
	idxs.push_back(28);
	idxs.push_back(30);
	idxs.push_back(31);
	idxs.push_back(46);
	idxs.push_back(48);


	// idxs.push_back(28);
	// idxs.push_back(29);

	std::random_device mRD;

	std::mt19937 mMT(mRD());
	std::uniform_real_distribution<double> mDistribution(0, 0.5);

	MultilevelSpline* s = new MultilevelSpline(1, mPhaseLength);
	s->SetKnots(0, mKnots);
	std::vector<Eigen::VectorXd> cps;
	for(int i = 0; i < mKnots.size(); i++) {
		Eigen::VectorXd cp(idxs.size());
		for(int j = 0; j < idxs.size(); j++) {
			// double d = cps_d[i][idxs[j]] - mPrevCps[i][idxs[j]];
			// d = d * mDistribution(mMT);
			// if(j < 6) 
			// 	d = 0;
			// cp(j) = mPrevCps[i][idxs[j]] + std::max(std::min(d, 0.3), -0.3);
			double d = mDistribution(mMT);
			// if(idxs[j] == 48 || idxs[j] == 46)
			// 	d *= 0.5;
			cp(j) = mPrevCps[i][idxs[j]] + mPrevCps[i][idxs[j]] ;

		}
		cps.push_back(cp);
	}

	s->SetControlPoints(0, cps);
	std::vector<Eigen::VectorXd> random_displacement = s->ConvertSplineToMotion();
	delete s;

	std::vector<Eigen::VectorXd> newpos;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < random_displacement.size(); i++) {

		Eigen::VectorXd p_bvh = mMotions_phase[i]->GetPosition();
		Eigen::VectorXd d = random_displacement[i];
		Eigen::VectorXd p = mMotions_phase_adaptive[i]->GetPosition();

		for(int j = 0; j < idxs.size(); j++) {
			int idx = idxs[j];
			p(idx) = d(j) + p_bvh(idx);
		}
		newpos.push_back(p);
	}

	std::vector<Eigen::VectorXd> newvel = this->GetVelocityFromPositions(newpos);
	std::vector<Motion*> motions_phase_temp;

	for(int i = 0; i < mMotions_phase_adaptive.size(); i++) {
		motions_phase_temp.push_back(new Motion(newpos[i], newvel[i]));
	}
	this->GenerateMotionsFromSinglePhase(1000, false, motions_phase_temp, mMotions_gen_temp[i]);

	while(!motions_phase_temp.empty()){
		Motion* m = motions_phase_temp.back();
		motions_phase_temp.pop_back();

		delete m;
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
		Eigen::VectorXd p_bvh = this->GetPosition(position[i].second);
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
	double rewardTarget = 0;
	double rewardTrajectory = 0;
    int mu = 60;
    std::cout << "num sample: " << mSamples.size() << std::endl;
    if(mSamples.size() < 300)
    	return false;

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
//	ofs.open(path, std::fstream::out | std::fstream::app);

	for(int i = 0; i < mu; i++) {
		double w = log(mu + 1) - log(i + 1);
	    weight_sum += w;
	    std::vector<Eigen::VectorXd> cps = std::get<0>(mSamples[i])->GetControlPoints(0);
	    for(int j = 0; j < num_knot; j++) {
			mean_cps[j] += w * cps[j].head(cps[j].rows() - 1);
	    }
	    rewardTrajectory += w * std::get<1>(mSamples[i]);
	    rewardTarget += std::get<2>(mSamples[i]);
	 //   ofs << mSamples[i].second << " ";
	}
	//ofs << std::endl;
	//ofs.close();

	rewardTrajectory /= weight_sum;
	rewardTarget /= (double)mu;

	std::cout << "current avg elite similarity reward: " << rewardTrajectory << ", target reward: " << rewardTarget << ", cutline: " << mPrevRewardTrajectory << std::endl;
	
	// if(mPrevRewardTrajectory < rewardTrajectory) {

		for(int i = 0; i < num_knot; i++) {
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
		return true;
	// } else {
	// 	while(mSamples.size() > 100){
	// 		MultilevelSpline* s = mSamples.back().first;
	// 		mSamples.pop_back();

	// 		delete s;
	// 	}
	// }
	return false;
}
};