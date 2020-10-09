#include "Controller.h"
#include "Character.h"
#include "MultilevelSpline.h"
#include <boost/filesystem.hpp>
#include <Eigen/QR>
#include <fstream>
#include <numeric>
#include <algorithm>
namespace DPhy
{	

Controller::Controller(ReferenceManager* ref, bool adaptive, bool record, int id)
	:mControlHz(30),mSimulationHz(600),mCurrentFrame(0),
	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25), w_srl(0.0),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{
	this->mRescaleParameter = std::make_tuple(1.0, 1.0, 1.0);
	this->isAdaptive = adaptive;
	this->mRecord = record;
	this->mReferenceManager = ref;
	this->id = id;
	this->mInputTargetParameters = mReferenceManager->GetTargetGoal();

	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();

	this->mWeight = 1.0;
	this->mGravity = Eigen::Vector3d(0,-9.81, 0);
	this->mWorld->setGravity(this->mGravity);

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml")).first;
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);
	
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());
	// SetSkeletonWeight(3.5);
	Eigen::VectorXd kp(this->mCharacter->GetSkeleton()->getNumDofs()), kv(this->mCharacter->GetSkeleton()->getNumDofs());

	kp.setZero();
	kv.setZero();
	this->mCharacter->SetPDParameters(kp,kv);
	mContacts.clear();
	mContacts.push_back("RightToe");
	mContacts.push_back("RightFoot");
	mContacts.push_back("LeftToe");
	mContacts.push_back("LeftFoot");

	mInterestedDof = mCharacter->GetSkeleton()->getNumDofs() - 6;
	mRewardDof = mCharacter->GetSkeleton()->getNumDofs();

	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	this->mCGL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("LeftFoot"));
	this->mCGR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("RightFoot"));
	this->mCGEL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("LeftToe"));
	this->mCGER = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("RightToe"));
	this->mCGHL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("LeftHand"));
	this->mCGHR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("RightHand"));
	this->mCGG = collisionEngine->createCollisionGroup(this->mGround.get());

	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	this->mMask.resize(dof);
	this->mMask.setZero();	
	mActions = Eigen::VectorXd::Zero(mInterestedDof + 1);
	mActions.setZero();

	mEndEffectors.clear();
	mEndEffectors.push_back("RightFoot");
	mEndEffectors.push_back("LeftFoot");
	mEndEffectors.push_back("LeftHand");
	mEndEffectors.push_back("RightHand");
	mEndEffectors.push_back("Head");


	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	this->mPDTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mPDTargetVelocities = Eigen::VectorXd::Zero(dof);

	//temp
	this->mRewardParts.resize(7, 0.0);
	targetParameters.resize(mReferenceManager->GetTargetBase().rows());
	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();

	ClearRecord();
	
	this->mHindsightCharacter.clear();
	this->mHindsightTarget.clear();
	this->mHindsightPhase.clear();

	mRewardLabels.clear();
	if(isAdaptive) {
		mRewardLabels.push_back("total_d");
		mRewardLabels.push_back("total_s");
		mRewardLabels.push_back("p");
		mRewardLabels.push_back("com");
		mRewardLabels.push_back("ee");
	} else {
		mRewardLabels.push_back("total");
		mRewardLabels.push_back("p");
		mRewardLabels.push_back("com");
		mRewardLabels.push_back("ee");
		mRewardLabels.push_back("v");
		mRewardLabels.push_back("time");
	}

	this->mIsHindsight = false;
	mSigTarget = 1;
}
const dart::dynamics::SkeletonPtr& 
Controller::GetSkeleton() { 
	return this->mCharacter->GetSkeleton(); 
}
void 
Controller::
Step()
{			


	if(IsTerminalState())
		return;
	Eigen::VectorXd s = this->GetState();
	Eigen::VectorXd a = mActions;

	// set action target pos
	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	for(int i = 0; i < mInterestedDof; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.2, -0.7*M_PI, 0.7*M_PI);
	}
	int sign = 1;
	if(mActions[mInterestedDof] < 0)
		sign = -1;
	double st = mActions[mInterestedDof];
	mActions[mInterestedDof] = (exp(abs(mActions[mInterestedDof])*3-2) - exp(-2)) * sign;
	mActions[mInterestedDof] = dart::math::clip(mActions[mInterestedDof], -0.8, 0.8);
	mAdaptiveStep = mActions[mInterestedDof];
	mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame += (1 + mAdaptiveStep);
	this->mCurrentFrameOnPhase += (1 + mAdaptiveStep);
	nTotalSteps += 1;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	
	// if(mRecord)
	// 	std::cout << mCurrentFrameOnPhase << " "<< mAdaptiveStep << " "<< mReferenceManager->GetTimeStep(mPrevFrameOnPhase, true) << std::endl;
	
	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame, isAdaptive);
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = mCharacter->GetSkeleton()->getPositionDifferences(mTargetPositions, mPrevTargetPositions) / 0.033;
	delete p_v_target;

	p_v_target = mReferenceManager->GetMotion(mCurrentFrame, false);
	this->mPDTargetPositions = p_v_target->GetPosition();
	this->mPDTargetVelocities = p_v_target->GetVelocity();
	
	delete p_v_target;

	int count_dof = 0;

	for(int i = 1; i <= num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getIndexInSkeleton(0);
		int dof = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getNumDofs();
		mPDTargetPositions.block(idx, 0, dof, 1) += mActions.block(count_dof, 0, dof, 1);
		count_dof += dof;

	}

	// set pd gain action
	Eigen::VectorXd kp(mCharacter->GetSkeleton()->getNumDofs()), kv(mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();

	for(int i = 1; i <= num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getIndexInSkeleton(0);
		int dof = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getNumDofs();
		std::string name = mCharacter->GetSkeleton()->getBodyNode(i)->getName();
		if(name.compare("Spine")==0){
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(1000);
		}
		else{
			if(dof == 3)
				kp.segment<3>(idx) = Eigen::Vector3d::Constant(500);
			else
				kp(idx) = 500;
		}
	}

	// KV_RATIO from CharacterConfiguration.h
	kv = KV_RATIO * kp;
	mCharacter->SetPDParameters(kp, kv);
	Eigen::VectorXd torque;
	Eigen::Vector3d d = Eigen::Vector3d(0, 0, 1);
	double end_f_sum = 0;	
	
	for(int i = 0; i < this->mSimPerCon; i += 2){
		torque = mCharacter->GetSPDForces(mPDTargetPositions, mPDTargetVelocities);
		for(int j = 0; j < num_body_nodes; j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			std::string name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
			double torquelim = mCharacter->GetTorqueLimit(name);
			double torque_norm = torque.block(idx, 0, dof, 1).norm();
		
			torque.block(idx, 0, dof, 1) = std::max(-torquelim, std::min(torquelim, torque_norm)) * torque.block(idx, 0, dof, 1).normalized();
		}
		for(int j = 0; j < 2; j++)
		{
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step(false);
		}
		mTimeElapsed += 2 * (1 + mAdaptiveStep);
	}
	 if(mCurrentFrameOnPhase >= 17 && mCurrentFrameOnPhase <= 64) {
		Eigen::Vector3d COM =  mCharacter->GetSkeleton()->getCOM();
		Eigen::Vector6d V = mCharacter->GetSkeleton()->getCOMSpatialVelocity();

		Eigen::Vector3d momentum;
		momentum.setZero();
		for(int i = 0; i < mCharacter->GetSkeleton()->getNumBodyNodes(); i++) {
			auto bn = mCharacter->GetSkeleton()->getBodyNode(i);
			Eigen::Matrix3d R = bn->getWorldTransform().linear();
			double Ixx, Iyy, Izz, Ixy, Ixz, Iyz;
			bn->getMomentOfInertia(Ixx, Iyy, Izz, Ixy, Ixz, Iyz);
			Eigen::Matrix3d I;
			I << Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz;
			I = R * I * R.transpose();
			Eigen::AngleAxisd aa(I); 
			Eigen::Vector3d aa_v = aa.axis() * aa.angle();
			momentum += aa_v + bn->getMass() * (bn->getCOM() - COM).cross(bn->getCOMLinearVelocity());
		}
		mVelocity += V.segment<3>(0);
		mMomentum += momentum;
		mCountTarget += 1;
		std::cout << this->mCurrentFrameOnPhase << " : " << V.segment<3>(0).transpose() << std::endl;
	}

	if(this->mCurrentFrameOnPhase > mReferenceManager->GetPhaseLength()){
		this->mCurrentFrameOnPhase -= mReferenceManager->GetPhaseLength();
		
		double weight = mCurrentFrameOnPhase / (1 + mAdaptiveStep);
		double f = mCurrentFrame - std::fmod(mCurrentFrame, mReferenceManager->GetPhaseLength());
		mHeadRoot = mReferenceManager->GetPosition(f, true).segment<6>(0);

		if(isAdaptive) {
			mTrackingRewardTrajectory /= mCountTracking;
			mReferenceManager->SaveTrajectories(data_spline, std::pair<double, double>(mTrackingRewardTrajectory, mTargetRewardTrajectory), targetParameters);
			data_spline.clear();
			mTrackingRewardTrajectory = 0;
			mTargetRewardTrajectory = 0;

			mControlFlag.setZero();
			mVelocity.setZero();
			mMomentum.setZero();
			mCountTarget = 0;
			mCountTracking = 0;
			
			if(mIsHindsight) {
				// to get V(t+1)
				mHindsightPhase.push_back(std::tuple<Eigen::VectorXd, Eigen::VectorXd, double>
									(mCharacter->GetSkeleton()->getPositions(), mCharacter->GetSkeleton()->getVelocities(), mCurrentFrame));
				mHindsightSAPhase.push_back(std::pair<Eigen::VectorXd, Eigen::VectorXd>(s, a));

				mHindsightCharacter.push_back(mHindsightPhase);
				mHindsightSA.push_back(mHindsightSAPhase);
				mHindsightTarget.push_back(targetParameters);
				
				mHindsightPhase.clear();
				mHindsightSAPhase.clear();
			}
		}
	}
	if(isAdaptive) {
		this->UpdateRewardTrajectory();
		this->UpdateAdaptiveReward();

	}
	else
		this->UpdateReward();

	this->UpdateTerminalInfo();

	if(mRecord) {
		SaveStepInfo();
	}

	if(isAdaptive)
	{
		data_spline.push_back(std::pair<Eigen::VectorXd,double>(mCharacter->GetSkeleton()->getPositions(), mCurrentFrame));
		if(mIsHindsight) {
			mHindsightPhase.push_back(std::tuple<Eigen::VectorXd, Eigen::VectorXd, double>
										(mCharacter->GetSkeleton()->getPositions(), mCharacter->GetSkeleton()->getVelocities(), mCurrentFrame));
			mHindsightSAPhase.push_back(std::pair<Eigen::VectorXd, Eigen::VectorXd>(s, a));
		}
	}

	mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;
	
	if(isAdaptive && mIsTerminal)
		data_spline.clear();

}
void
Controller::
SaveStepInfo() 
{
	mRecordBVHPosition.push_back(mReferenceManager->GetPosition(mCurrentFrame, false));
	mRecordTargetPosition.push_back(mTargetPositions);
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	mRecordCOM.push_back(mCharacter->GetSkeleton()->getCOM());
	mRecordPhase.push_back(mCurrentFrame);
	
	bool rightContact = CheckCollisionWithGround("RightFoot") || CheckCollisionWithGround("RightToe");
	bool leftContact = CheckCollisionWithGround("LeftFoot") || CheckCollisionWithGround("LeftToe");

	mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));
}
void 
Controller::
ClearRecord() 
{
	this->mRecordVelocity.clear();
	this->mRecordPosition.clear();
	this->mRecordCOM.clear();
	this->mRecordTargetPosition.clear();
	this->mRecordBVHPosition.clear();
	this->mRecordObjPosition.clear();
	this->mRecordPhase.clear();
	this->mRecordFootContact.clear();
	this->mRecordTorqueNorm.clear();

	this->mHindsightPhase.clear();
	this->mHindsightSAPhase.clear();

	this->mControlFlag.resize(4);
	this->mControlFlag.setZero();

	mCountTarget = 0;
	mCountTracking = 0;
	mVelocity.setZero();
	mMomentum.setZero();
	data_spline.clear();

}

std::vector<double> 
Controller::
GetTrackingReward(Eigen::VectorXd position, Eigen::VectorXd position2, 
	Eigen::VectorXd velocity, Eigen::VectorXd velocity2, std::vector<std::string> list, bool useVelocity)
{
	auto& skel = this->mCharacter->GetSkeleton();
	int dof = skel->getNumDofs();
	int num_body_nodes = skel->getNumBodyNodes();

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	Eigen::VectorXd p_diff = skel->getPositionDifferences(position, position2);
	Eigen::VectorXd p_diff_reward;
	
	p_diff_reward = p_diff;
	if(isAdaptive)
		p_diff_reward.segment<3>(0) *= 5;
	Eigen::VectorXd v_diff, v_diff_reward;

	if(useVelocity) {
		v_diff = skel->getVelocityDifferences(velocity, velocity2);
		v_diff_reward = v_diff;
	}

	skel->setPositions(position);
	skel->computeForwardKinematics(true,false,false);

	std::vector<Eigen::Isometry3d> ee_transforms;
	Eigen::VectorXd ee_diff(mEndEffectors.size()*3);
	ee_diff.setZero();	
	for(int i=0;i<mEndEffectors.size(); i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
	}
	
	Eigen::Vector3d com_diff = skel->getCOM();
	
	skel->setPositions(position2);
	skel->computeForwardKinematics(true,false,false);

	for(int i=0;i<mEndEffectors.size();i++){
		Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee_diff.segment<3>(3*i) = diff.translation();
	}
	com_diff -= skel->getCOM();

	double scale = 1.0;

	double sig_p = 0.4 * scale; 
	double sig_v = 3 * scale;	
	double sig_com = 0.2 * scale;		
	double sig_ee = 0.2 * scale;		

	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_v;
	if(useVelocity)
	{
		r_v = exp_of_squared(v_diff_reward,sig_v);
	}
	double r_ee = exp_of_squared(ee_diff,sig_ee);
	double r_com = exp_of_squared(com_diff,sig_com);

	std::vector<double> rewards;
	rewards.clear();

	rewards.push_back(r_p);
	rewards.push_back(r_com);
	rewards.push_back(r_ee);

	if(useVelocity) {
		rewards.push_back(r_v);
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	return rewards;

}
double 
Controller::
GetTargetReward()
{
	double r_target = 0;
	auto& skel = this->mCharacter->GetSkeleton();
	if(mCurrentFrameOnPhase >= 64 && mControlFlag[0] == 0) {
		mVelocity /= mCountTarget;
		Eigen::Vector3d v_diff = (mVelocity - mInputTargetParameters.segment<3>(1)) * 5;
		r_target = exp_of_squared(v_diff, mSigTarget);
		if(mRecord) {
		 	std::cout << mVelocity.transpose() << " " << v_diff.transpose() << " "<<  r_target << std::endl;
		}
		mControlFlag[0] = 1;		

	} 
	return r_target;
}
std::vector<bool> 
Controller::
GetContacts() 
{
	auto& skel = this->mCharacter->GetSkeleton();

	std::vector<bool> result;
	result.clear();
	for(int i = 0; i < mContacts.size(); i++) {
		Eigen::Vector3d p = skel->getBodyNode(mContacts[i])->getWorldTransform().translation();
		if(p[1] < 0.04) {
			result.push_back(true);
		} else {
			result.push_back(false);
		}
	}

	return result;
}
std::vector<bool> 
Controller::
GetContacts(Eigen::VectorXd pos) 
{
	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	
	skel->setPositions(pos);
	skel->computeForwardKinematics(true,false,false);

	std::vector<bool> result;
	result.clear();
	for(int i = 0; i < mContacts.size(); i++) {
		Eigen::Vector3d p = skel->getBodyNode(mContacts[i])->getWorldTransform().translation();
		if(p[1] < 0.07) {
			result.push_back(true);
		} else {
			result.push_back(false);
		}
	}
	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	return result;
}
void
Controller::
UpdateAdaptiveReward()
{

	auto& skel = this->mCharacter->GetSkeleton();
	
	std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), mTargetPositions,
								 skel->getVelocities(), mTargetVelocities, mRewardBodies, false);
	double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();
	double r_t = this->GetTargetReward();
	
	std::vector<std::pair<bool, Eigen::Vector3d>> contacts_ref = mReferenceManager->GetContactInfo(mReferenceManager->GetPosition(mCurrentFrameOnPhase, false));
	std::vector<std::pair<bool, Eigen::Vector3d>> contacts_cur = mReferenceManager->GetContactInfo(skel->getPositions());
	double con_diff = 0;

	for(int i = 0; i < contacts_cur.size(); i++) {
		if(contacts_ref[i].first || contacts_cur[i].first) {
			con_diff += pow(((contacts_cur[i].second)(1) - (contacts_ref[i].second)(1)) * 5, 2);
		}
	}
	double r_con = exp(-con_diff);
	double time_diff = (mAdaptiveStep + 1) - mReferenceManager->GetTimeStep(mPrevFrameOnPhase, true);
	double r_time = exp(-pow(time_diff, 2)*75);

	double r_tot = 0.8 * accum_bvh + 0.1 * r_con + 0.1 * r_time;
	// if(mCurrentFrameOnPhase >= 30 && mCurrentFrameOnPhase <= 45) {
	// 	double r_max = 0;
	// 	double p_max = 0;
	// 	for(int i = 0; i <= 20; i++) {
	// 		double p = mCurrentFrame - 2 + 0.2 * i;
	// 		Motion* p_v_target = mReferenceManager->GetMotion(p, isAdaptive);
	// 		Eigen::VectorXd p_temp = p_v_target->GetPosition();
	// 		Eigen::VectorXd v_temp = p_v_target->GetVelocity();
	// 		delete p_v_target;

	// 		std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), p_temp,
	// 									 skel->getVelocities(), v_temp, mRewardBodies, false);
	// 		double accum = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();
	// 		if(accum > r_max) {
	// 			p_max = p;
	// 			r_max = accum;
	// 		}
	// 	}
	// 	std::cout << mCurrentFrameOnPhase << " " << accum_bvh << ", " << p_max << " " << r_max << std::endl;
	// }

	mRewardParts.clear();
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(10 * r_t);
		mRewardParts.push_back(tracking_rewards_bvh[0]);
		mRewardParts.push_back(tracking_rewards_bvh[1]);
		mRewardParts.push_back(tracking_rewards_bvh[2]);
	}
	if(r_t != 0) {
		mTargetRewardTrajectory += r_t;
	}
	mTrackingRewardTrajectory += r_tot; //(0.4 * tracking_rewards_bvh[0] + 0.4 * tracking_rewards_bvh[1] + 0.2 * r_con);
	mCountTracking += 1;
}
void
Controller::
UpdateReward()
{
	auto& skel = this->mCharacter->GetSkeleton();
	std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), mTargetPositions,
								 skel->getVelocities(), mTargetVelocities, mRewardBodies, true);
	double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();

	double r_time = exp(-pow(mActions[mInterestedDof],2)*50);
	mRewardParts.clear();
	double r_tot = 0.9 * (0.5 * tracking_rewards_bvh[0] + 0.1 * tracking_rewards_bvh[1] + 0.3 * tracking_rewards_bvh[2] + 0.1 * tracking_rewards_bvh[3] ); // + 0.1 * r_time;
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(tracking_rewards_bvh[0]);
		mRewardParts.push_back(tracking_rewards_bvh[1]);
		mRewardParts.push_back(tracking_rewards_bvh[2]);
		mRewardParts.push_back(tracking_rewards_bvh[3]);
		mRewardParts.push_back(r_time);
	}
}
void 
Controller::
UpdateRewardTrajectory() {
	// std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), mTargetPositions,
	// 							 skel->getVelocities(), mTargetVelocities, mRewardBodies, false);

	// mTrackingRewardTrajectory += (0.5 * tracking_rewards_bvh[0] + 0.5 * tracking_rewards_bvh[2]);

}
std::vector<std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>>> 
Controller::
GetHindsightSAR(std::vector<std::vector<Eigen::VectorXd>> cps)
{
	std::cout << "hindsight" << std::endl;
	std::vector<double> count;
	for(int i = 0; i < cps.size(); i++) {
		count.push_back(mHindsightCharacter[i].size());
	}
	// std::cout << count.size() << " ";
	// for(int i = 0; i < count.size(); i++) {
	// 	std::cout << count[i] << " ";
	// }
	DPhy::MultilevelSpline* s = new DPhy::MultilevelSpline(1, mReferenceManager->GetPhaseLength());
	s->SetKnots(0, mReferenceManager->GetKnots());

	std::vector<std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>>> sar;

	std::vector<Motion*> motions_cps;
	std::vector<Motion*> motions_phase_cps;
	for(int i = 0; i < cps.size(); i++) {

		auto cps_phase = cps[i]; 
		s->SetControlPoints(0, cps_phase);

		std::vector<Eigen::VectorXd> newpos;
		std::vector<Eigen::VectorXd> new_displacement = s->ConvertSplineToMotion();
		mReferenceManager->AddDisplacementToBVH(new_displacement, newpos);
		std::vector<Eigen::VectorXd> newvel = mReferenceManager->GetVelocityFromPositions(newpos);

		for(int j = 0; j < newpos.size(); j++) {
			if(motions_phase_cps.size() <= j)
				motions_phase_cps.push_back(new Motion(newpos[j], newvel[j]));
			else {
				motions_phase_cps[j]->SetPosition(newpos[j]);
				motions_phase_cps[j]->SetVelocity(newvel[j]);
			}
		}
		
		int len = std::ceil(std::get<2>(mHindsightCharacter[i].back())) + 10;
		mReferenceManager->GenerateMotionsFromSinglePhase(len, false, motions_phase_cps, motions_cps);

		std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>> sar_epi;
		mInputTargetParameters = mHindsightTarget[i];
		
		mControlFlag[0] = 0;

		auto skel = mCharacter->GetSkeleton();
		for(int j = 0; j < mHindsightCharacter[i].size() - 1; j++) {
			mCurrentFrame = std::get<2>(mHindsightCharacter[i][j]);
			mCurrentFrameOnPhase = std::fmod(mCurrentFrame, mReferenceManager->GetPhaseLength());
			
			skel->setPositions(std::get<0>(mHindsightCharacter[i][j]));
			skel->setVelocities(std::get<1>(mHindsightCharacter[i][j]));
			skel->computeForwardKinematics(true,true,false);
			
			int k0 = (int) std::floor(mCurrentFrame);
			int k1 = (int) std::ceil(mCurrentFrame);	

			mTargetPositions = DPhy::BlendPosition(motions_cps[k1]->GetPosition(), motions_cps[k0]->GetPosition(), 1 - (mCurrentFrame-k0));
			mTargetVelocities = DPhy::BlendPosition(motions_cps[k1]->GetVelocity(), motions_cps[k0]->GetVelocity(), 1 - (mCurrentFrame-k0));
			
			this->UpdateAdaptiveReward();

			Eigen::VectorXd rewards(2);
			rewards << mRewardParts[0], mRewardParts[1];

			(mHindsightSA[i][j].first).tail(mInputTargetParameters.rows()) = mInputTargetParameters;
			sar_epi.push_back(std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>
				(mHindsightSA[i][j].first, mHindsightSA[i][j].second, rewards, mCurrentFrameOnPhase));
		}
	
		sar_epi.push_back(std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>
				(mHindsightSA[i][mHindsightCharacter[i].size() - 1].first, mHindsightSA[i][mHindsightCharacter[i].size() - 1].second,
				 Eigen::Vector2d(0, 0), 0));
		sar.push_back(sar_epi);
	}
	// std::cout << count.size() << " ";
	// for(int i = 0; i < count.size(); i++) {
	// 	std::cout << count[i] << " ";
	// }
	while(!motions_cps.empty()){
		Motion* m = motions_cps.back();
		motions_cps.pop_back();

		delete m;
	}	
	while(!motions_phase_cps.empty()){
		Motion* m = motions_phase_cps.back();
		motions_phase_cps.pop_back();

		delete m;
	}		

	delete s;
	
	mHindsightSA.clear();
	mHindsightTarget.clear();
	mHindsightCharacter.clear();

	return sar;
}
void
Controller::
UpdateTerminalInfo()
{	
	Eigen::VectorXd p_ideal = mTargetPositions;
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(mTargetPositions);
	skel->computeForwardKinematics(true,false,false);

	Eigen::Isometry3d root_diff = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	
	Eigen::AngleAxisd root_diff_aa(root_diff.linear());
	double angle = RadianClamp(root_diff_aa.angle());
	Eigen::Vector3d root_pos_diff = root_diff.translation();


	// check nan
	if(dart::math::isNan(p)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 3;
	}
	if(dart::math::isNan(v)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 4;
	}
	//characterConfigration
	if(root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 2;
	}
	if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
		mIsTerminal = true;
		terminationReason = 1;
	}
	else if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 5;
	}
	else if(!mRecord && nTotalSteps > mReferenceManager->GetPhaseLength()* 6 + 10) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}
	if(mRecord) {
		if(mIsTerminal) std::cout << terminationReason << std::endl;
	}

	if(mIsTerminal && terminationReason != 8)
		mReferenceManager->ReportEarlyTermination();

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

}
bool
Controller::
FollowBvh()
{	
	if(IsTerminalState())
		return false;
	auto& skel = mCharacter->GetSkeleton();

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	mTargetPositions = p_v_target->GetPosition();
	mTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	for(int i=0;i<this->mSimPerCon;i++)
	{
		skel->setPositions(mTargetPositions);
		skel->setVelocities(mTargetVelocities);
		skel->computeForwardKinematics(true, true, false);
	}
	this->mCurrentFrame += 1;
	this->nTotalSteps += 1;
	return true;
}
void
Controller::
SetSkeletonWeight(double weight)
{
	double w = weight / mWeight;
	mWeight = weight;

	std::vector<std::tuple<std::string, Eigen::Vector3d, double>> deform;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < n_bnodes; i++){
		std::string name = mCharacter->GetSkeleton()->getBodyNode(i)->getName();
		// if(name.find("Shoulder") != std::string::npos ||
		//    name.find("Arm") != std::string::npos ||
		//    name.find("Hand") != std::string::npos) {
		// 	deform.push_back(std::make_tuple(name, Eigen::Vector3d(1, 1, 1), w));
		// }

		if(name.find("Leg") != std::string::npos ||
		   name.find("Foot") != std::string::npos ||
		   name.find("Toe") != std::string::npos) {
			deform.push_back(std::make_tuple(name, Eigen::Vector3d(1, 1, 1), w));
		}
	}

	DPhy::SkeletonBuilder::DeformSkeleton(mCharacter->GetSkeleton(), deform);
	std::cout << "current weight : "  << mWeight << ", " << mCharacter->GetSkeleton()->getMass() << std::endl;

}
void 
Controller::
Reset(bool RSI)
{
	this->mWorld->setGravity(this->mGravity);
	this->mWorld->reset();
	auto& skel = mCharacter->GetSkeleton();
	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();
	//RSI
	if(RSI && !isAdaptive) {
		this->mCurrentFrame = (int) dart::math::Random::uniform(0.0, mReferenceManager->GetPhaseLength()-5.0);
	}
	else {
		this->mCurrentFrame = 0; // 0;
		this->mTargetRewardTrajectory = 0;
		this->mTrackingRewardTrajectory = 0;
	}

	this->mCurrentFrameOnPhase = this->mCurrentFrame;
	this->mStartFrame = this->mCurrentFrame;
	this->nTotalSteps = 0;
	this->mTimeElapsed = 0;

	Motion* p_v_target;
	p_v_target = mReferenceManager->GetMotion(mCurrentFrame, isAdaptive);

	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	this->mPDTargetPositions = mTargetPositions;
	this->mPDTargetVelocities = mTargetVelocities;

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;

	ClearRecord();
	SaveStepInfo();

	mHeadRoot = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;
	
	if(isAdaptive)
	{
		data_spline.push_back(std::pair<Eigen::VectorXd,double>(mCharacter->GetSkeleton()->getPositions(), mCurrentFrame));
	}
}
int
Controller::
GetNumState()
{
	return this->mNumState;
}
int
Controller::
GetNumAction()
{
	return this->mNumAction;
}
void 
Controller::
SetAction(const Eigen::VectorXd& action)
{
	this->mActions = action;
}
Eigen::VectorXd 
Controller::
GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel) {
	Eigen::VectorXd ret;
	auto& skel = mCharacter->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	// int num_body_nodes = mInterestedBodies.size();
	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(pos);
	skel->setVelocities(vel);
	skel->computeForwardKinematics(true, true, false);

//	Eigen::Isometry3d target_root_inv = root->getWorldTransform().inverse();

	ret.resize((num_ee)*9+12);
	for(int i=0;i<num_ee;i++)
	{		
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
//		Eigen::Isometry3d transform = target_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
		ret.segment<6>(6*i) << rot, transform.translation();
	}


	for(int i=0;i<num_ee;i++)
	{
	    int idx = skel->getBodyNode(mEndEffectors[i])->getParentJoint()->getIndexInSkeleton(0);
//		ret.segment<3>(6*num_ee + 3*i) << target_root_inv.linear() * vel.segment<3>(idx);
	    ret.segment<3>(6*num_ee + 3*i) << vel.segment<3>(idx);

	}

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
	Eigen::Vector3d root_angular_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getAngularVelocity();
	Eigen::Vector3d root_linear_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getCOMLinearVelocity();

//	ret.tail<9>() << rot, root_angular_vel_relative, root_linear_vel_relative;
	ret.tail<12>() << rot, transform.translation(), root_angular_vel_relative, root_linear_vel_relative;

	// restore
	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true, true, false);

	return ret;
}
bool
Controller::
CheckCollisionWithGround(std::string bodyName){
	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	dart::collision::CollisionOption option;
	dart::collision::CollisionResult result;
	if(bodyName == "RightFoot"){
		bool isCollide = collisionEngine->collide(this->mCGR.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftFoot"){
		bool isCollide = collisionEngine->collide(this->mCGL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "RightToe"){
		bool isCollide = collisionEngine->collide(this->mCGER.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftToe"){
		bool isCollide = collisionEngine->collide(this->mCGEL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "RightHand"){
		bool isCollide = collisionEngine->collide(this->mCGHR.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftHand"){
		bool isCollide = collisionEngine->collide(this->mCGHL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else{ // error case
		std::cout << "check collision : bad body name" << std::endl;
		return false;
	}
}
Eigen::VectorXd 
Controller::
GetState()
{
	if(mIsTerminal && terminationReason != 8){
		return Eigen::VectorXd::Zero(mNumState);
	}
	auto& skel = mCharacter->GetSkeleton();
	
	double root_height = skel->getRootBodyNode()->getCOM()[1];

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	Eigen::VectorXd p,v;
	p.resize(p_save.rows()-6);
	p = p_save.tail(p_save.rows()-6);
	v = v_save; ///10.0;

	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();
	Eigen::VectorXd ee;
	ee.resize(mEndEffectors.size()*3);
	for(int i=0;i<mEndEffectors.size();i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee.segment<3>(3*i) << transform.translation();
	}

	double t = mReferenceManager->GetTimeStep(mCurrentFrameOnPhase, isAdaptive);
	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame+t, isAdaptive);
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->GetPosition(), p_v_target->GetVelocity()*t);
	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	double phase = ((int) mCurrentFrame % mReferenceManager->GetPhaseLength()) / (double) mReferenceManager->GetPhaseLength();
	Eigen::VectorXd state;

	double com_diff = 0;

	if(isAdaptive) {
		state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+1); //+mInputTargetParameters.rows());
		state<< p, v, up_vec_angle, root_height, p_next, ee, mCurrentFrameOnPhase; //, mInputTargetParameters;
	}
	else {
		state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+1);
		state<< p, v, up_vec_angle, root_height, p_next, ee, mCurrentFrameOnPhase;
	}
	return state;
}
void
Controller::SaveTimeData(std::string directory) {
	std::string path = std::string(CAR_DIR) + std::string("/") +  directory;
	std::cout << "save results to" << path << std::endl;
	
	std::ofstream ofs(path);
	ofs << mReferenceManager->GetPhaseLength() << std::endl;

	for(int i = 0; i < mRecordPhase.size() - 1; i++) {
		ofs << i << " " << mRecordPhase[i] << " " << mRecordPhase[i+1] - mRecordPhase[i] << std::endl;
	}
	ofs.close();

}
void
Controller::SaveDisplayedData(std::string directory, bool bvh) {
	std::string path = std::string(CAR_DIR) + std::string("/") +  directory;
	std::cout << "save results to" << path << std::endl;

	std::ofstream ofs(path);
	std::vector<std::string> bvh_order;
	bvh_order.push_back("Hips");
	bvh_order.push_back("Spine");
	bvh_order.push_back("Spine1");
	bvh_order.push_back("Spine2");
	bvh_order.push_back("LeftShoulder");
	bvh_order.push_back("LeftArm");
	bvh_order.push_back("LeftForeArm");
	bvh_order.push_back("LeftHand");
	bvh_order.push_back("RightShoulder");
	bvh_order.push_back("RightArm");
	bvh_order.push_back("RightForeArm");
	bvh_order.push_back("RightHand");
	bvh_order.push_back("Neck");
	bvh_order.push_back("Head");
	bvh_order.push_back("LeftUpLeg");
	bvh_order.push_back("LeftLeg");
	bvh_order.push_back("LeftFoot");
	bvh_order.push_back("LeftToe");
	bvh_order.push_back("RightUpLeg");
	bvh_order.push_back("RightLeg");
	bvh_order.push_back("RightFoot");
	bvh_order.push_back("RightToe");


	if(bvh)
		ofs << mRecordPosition.size() << std::endl;
	for(auto t: mRecordPosition) {
		if(bvh) {
			ofs << t.segment<3>(3).transpose() * 100 << " ";

			for(int i = 0; i < bvh_order.size(); i++) {
				int idx = mCharacter->GetSkeleton()->getBodyNode(bvh_order[i])->getParentJoint()->getIndexInSkeleton(0);
				Eigen::AngleAxisd aa(t.segment<3>(idx).norm(), t.segment<3>(idx).normalized());
				Eigen::Matrix3d m;
				m = aa;
				Eigen::Vector3d v = dart::math::matrixToEulerZXY(m);
				ofs << v.transpose() * 180 / M_PI << " ";			
			}
			ofs << std::endl;
		} else {
			ofs << t.transpose() << std::endl;
		}
		
	}
	std::cout << "saved position: " << mRecordPosition.size() << ", "<< mReferenceManager->GetPhaseLength() << ", " << mRecordPosition[0].rows() << std::endl;
	ofs.close();
}
}
