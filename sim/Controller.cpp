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

Controller::Controller(ReferenceManager* ref, bool adaptive, bool parametric, bool record, int id)
	:mControlHz(30),mSimulationHz(150),mCurrentFrame(0),
	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{
	this->mRescaleParameter = std::make_tuple(1.0, 1.0, 1.0);
	this->isAdaptive = adaptive;
	this->isParametric = parametric;
	this->mRecord = record;
	this->mReferenceManager = ref;
	this->id = id;
	this->mParamGoal = mReferenceManager->GetParamGoal();

	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();

	this->mBaseGravity = Eigen::Vector3d(0,-9.81, 0);
	this->mWorld->setGravity(this->mBaseGravity);

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml")).first;
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);
	
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());

	this->mBaseMass = mCharacter->GetSkeleton()->getMass();
	this->mMass = mBaseMass;

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
	mParamCur.resize(mReferenceManager->GetParamGoal().rows());
	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();

	ClearRecord();
	
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

	if(isAdaptive) {
		path = std::string(CAR_DIR)+std::string("/character/box.xml");
		this->mObject = new DPhy::Character(path);	
		this->mWorld->addSkeleton(this->mObject->GetSkeleton());
	}

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
	mActions[mInterestedDof] = (exp(abs(mActions[mInterestedDof])*0.25)-1) * sign;
	mActions[mInterestedDof] = dart::math::clip(mActions[mInterestedDof], -0.8, 0.8);
	mAdaptiveStep = mActions[mInterestedDof];
	// mAdaptiveStep = 0;

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

	Eigen::VectorXd torque;
	Eigen::Vector3d d = Eigen::Vector3d(0, 0, 1);
	double end_f_sum = 0;	
	
	for(int i = 0; i < this->mSimPerCon; i += 2){

		for(int j = 0; j < 2; j++) {
			mCharacter->GetSkeleton()->setSPDTarget(mPDTargetPositions, 600, 49);
			//Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
			// for(int j = 0; j < num_body_nodes; j++) {
			// 	int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
			// 	int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			// 	std::string name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
			// 	double torquelim = mCharacter->GetTorqueLimit(name);
			// 	double torque_norm = torque.block(idx, 0, dof, 1).norm();
			
			// 	torque.block(idx, 0, dof, 1) = std::max(-torquelim, std::min(torquelim, torque_norm)) * torque.block(idx, 0, dof, 1).normalized();
			// }

			//mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step(false);
		}

		mTimeElapsed += 2 * (1 + mAdaptiveStep);
		if(mControlFlag[0] >= 1 && mControlFlag[0] < 3) {
			if(mObject->GetSkeleton()->getBodyNode("Base1")->getCOMLinearVelocity().norm() > maxSpeedObj) {
				maxSpeedObj = mObject->GetSkeleton()->getBodyNode("Base1")->getCOMLinearVelocity().norm();
			}
		}

	}
	if(isAdaptive && mCurrentFrameOnPhase >= 17 && mControlFlag[0] == 0) {
		Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond( mCharacter->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform().linear()));
		rot = projectToXZ(rot);		
		Eigen::AngleAxisd obj_dir(rot.norm(), rot.normalized());
		Eigen::Vector3d obj_pos = mCharacter->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform().translation();
		Eigen::Vector3d delta(0.065 + 0.15 + 0.02, 0 , 0.02);
		delta = obj_dir * delta;
		Eigen::VectorXd p_obj(mObject->GetSkeleton()->getNumDofs());
			
		p_obj.setZero();

		for(int i = 0; i < mObject->GetSkeleton()->getNumBodyNodes(); i++) {
			std::string name = mObject->GetSkeleton()->getBodyNode(i)->getName();
			if(!name.compare("Sandbag"))
				continue;

			int idx = mObject->GetSkeleton()->getBodyNode(i)->getParentJoint()->getIndexInSkeleton(0);
			if(!name.compare("Ground")) {
				p_obj.segment<3>(idx) = obj_dir.angle() * obj_dir.axis();
				p_obj.segment<3>(idx + 3) = obj_pos - delta;
				p_obj[idx + 4] = 0;
			} else if (!name.compare("Base2")) {
				p_obj[idx] = obj_pos[1] - 0.9;
			}
		}

		mObject->GetSkeleton()->setPositions(p_obj);
		mObject->GetSkeleton()->setVelocities(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->setAccelerations(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->computeForwardKinematics(true,false,false);

		mControlFlag[0] = 1;

	} else if(isAdaptive && mControlFlag[0] == 1) {
		mHandPosition = mCharacter->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform().translation();
		Eigen::VectorXd p_obj(mObject->GetSkeleton()->getNumDofs());
		p_obj.setZero();
		p_obj.segment<3>(3) = Eigen::Vector3d(-2.0, 0.0, -2.0);
		mObject->GetSkeleton()->setPositions(p_obj);
		mObject->GetSkeleton()->setVelocities(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->setAccelerations(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->computeForwardKinematics(true,false,false);
		mControlFlag[0] = 2;
	} else if(mControlFlag[0] == 2) {
		mControlFlag[0] = 3;
	}

	if(mCountHead < 5) {
		mHeadRoot += mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		mCountHead += 1;
		if(mCountHead == 5) {
			mHeadRoot /= 5;
		}
	}
	if(this->mCurrentFrameOnPhase > mReferenceManager->GetPhaseLength()){
		this->mCurrentFrameOnPhase -= mReferenceManager->GetPhaseLength();
		mHeadRoot = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		mRootZero = mHeadRoot;
		mCountHead = 1;

		if(isAdaptive) {
			mTrackingRewardTrajectory /= mCountTracking;
			mTWRewardTrajectory /= mCountTracking;
			for(int i = 0; i < mRewardSimilarity.size(); i++) {
				mRewardSimilarity[i] /= mCountTracking;
			}
			mReferenceManager->SaveTrajectories(data_spline, std::tuple<double, double, std::vector<double>>(mTrackingRewardTrajectory, mParamRewardTrajectory, mRewardSimilarity), mParamCur);
			data_spline.clear();
			mTWRewardTrajectory = 0;
			mTrackingRewardTrajectory = 0;
			mParamRewardTrajectory = 0;
			mRewardSimilarity.clear();
			
			mControlFlag.setZero();
			mCountParam = 0;
			mCountTracking = 0;
			
			maxSpeedObj = 0;
			mHandPosition.setZero();
		}
	}
	if(isAdaptive) {
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
		data_spline.push_back(std::pair<Eigen::VectorXd,double>(mCharacter->GetSkeleton()->getPositions(), mCurrentFrameOnPhase));
	}

	mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;
	if(mPosQueue.size() >= 3)
		mPosQueue.pop();
	if(mTimeQueue.size() >= 3)
		mTimeQueue.pop();
	mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	mTimeQueue.push(mCurrentFrame);

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

	this->mControlFlag.resize(4);
	this->mControlFlag.setZero();

	mCountParam = 0;
	mCountTracking = 0;
	data_spline.clear();
	mRewardSimilarity.clear();

	maxSpeedObj = 0;
	mHandPosition.setZero();
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
std::vector<std::pair<bool, Eigen::Vector3d>> 
Controller::
GetContactInfo(Eigen::VectorXd pos) 
{
	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	
	skel->setPositions(pos);
	skel->computeForwardKinematics(true,false,false);

	std::vector<std::string> contact;
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
Controller::
GetSimilarityReward()
{
auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	auto p_v_target = mReferenceManager->GetMotion(mCurrentFrameOnPhase, false);
	Eigen::VectorXd pos = p_v_target->GetPosition();
	Eigen::VectorXd vel = p_v_target->GetVelocity();
	delete p_v_target;

	// std::cout<<mCurrentFrame<<"/ ref: "<<ref_obj_height<<" / cur: "<<cur_obj_height<<std::endl;

	std::vector<std::pair<bool, Eigen::Vector3d>> contacts_ref = GetContactInfo(pos);
	std::vector<std::pair<bool, Eigen::Vector3d>> contacts_cur = GetContactInfo(skel->getPositions());

	double con_diff = 0;

	for(int i = 0; i < contacts_cur.size(); i++) {
		if(contacts_ref[i].first || contacts_cur[i].first) {
			con_diff += pow(((contacts_cur[i].second)(1) - (contacts_ref[i].second)(1)) * 5, 2);
		}
	}
	//double r_con = exp(-con_diff);
	Eigen::VectorXd p_aligned = skel->getPositions();
	std::vector<Eigen::VectorXd> p_with_zero;
	p_with_zero.push_back(mRootZero);
	p_with_zero.push_back(p_aligned.segment<6>(0));
	p_with_zero = Align(p_with_zero, mReferenceManager->GetPosition(0, false));
	p_aligned.segment<6>(0) = p_with_zero[1];

	Eigen::VectorXd v = skel->getPositionDifferences(skel->getPositions(), mPosQueue.front()) / (mCurrentFrame - mTimeQueue.front() + 1e-10) / 0.033;
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

	Eigen::VectorXd p_diff = skel->getPositionDifferences(pos, p_aligned);
	Eigen::VectorXd v_diff = skel->getVelocityDifferences(vel, v);

	int num_body_nodes = skel->getNumBodyNodes();
	for(int i =0 ; i < vel.rows(); i++) {
		v_diff(i) = v_diff(i) / std::max(0.5, vel(i));
	}
	for(int i = 0; i < num_body_nodes; i++) {
		std::string name = mCharacter->GetSkeleton()->getBodyNode(i)->getName();
		int idx = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getIndexInSkeleton(0);
		if(name.compare("Hips") == 0 ) {
			p_diff.segment<3>(idx) *= 3;
			p_diff.segment<3>(idx + 3) *= 3;
			v_diff.segment<3>(idx) *= 3;
			v_diff.segment<3>(idx + 3) *= 3;
		} else if(name.find("Spine") != std::string::npos) {
			p_diff.segment<3>(idx) *= 3;
			v_diff.segment<3>(idx) *= 3;

		}
	}
	double r_con = exp(-con_diff);
	double r_ee = exp_of_squared(v_diff, 3);
	double r_p = exp_of_squared(p_diff,0.3);
	mPrevFrame2 = mPrevFrame;
	mPrevFrame = mCurrentFrame;

	if(mRewardSimilarity.size() == 0) {
		for(int i = 0; i < 3; i++) {
			mRewardSimilarity.push_back(0);
		}
	}
	
	mRewardSimilarity[0] += r_con;
	mRewardSimilarity[1] += p_diff.dot(p_diff) / p_diff.rows();
	mRewardSimilarity[2] += v_diff.dot(v_diff) / v_diff.rows();

	return exp(-r_con)  * r_p * r_ee;
}
double 
Controller::
GetParamReward()
{
	double r_param = 0;
	auto& skel = this->mCharacter->GetSkeleton();
	if(mControlFlag[0] == 3) {
		Eigen::Vector3d root_new = mHeadRoot.segment<3>(0);
		root_new = projectToXZ(root_new);
		Eigen::AngleAxisd aa(root_new.norm(), root_new.normalized());
		Eigen::Vector3d dir = Eigen::Vector3d(0.7, 0, - sqrt(1 - 0.49));
		dir.normalize();
		dir *= mParamGoal(0);
		Eigen::Vector3d goal_hand = aa * dir + mHeadRoot.segment<3>(3);
		goal_hand(1) = 1.3;
		Eigen::Vector3d hand_diff = goal_hand - mHandPosition;
		double v_diff = mParamGoal(1) - maxSpeedObj;

		r_param = exp_of_squared(hand_diff,0.1) * exp(-pow(v_diff, 2)*150);
		
		Eigen::Vector3d hand = mHandPosition;
		hand = hand - mHeadRoot.segment<3>(3);
		hand(1) = 0;
		dir = aa.inverse() * hand;
		double norm = dir.norm();
		dir.normalize();

		if(abs(0.7 - dir(0)) < 0.1 && abs(1.3 - mHandPosition(1)) < 0.05)
			mParamCur << norm, maxSpeedObj;
		else
			mParamCur << -1, -1;
		mControlFlag[0] = 4;

		if(mRecord) {
			std::cout << hand_diff.transpose() << " "<< exp_of_squared(hand_diff, 0.4)  << " "<< exp_of_squared(hand_diff,0.1) << std::endl;
			std::cout << v_diff << " "<< exp(-pow(v_diff, 2)*10)  << " "<< exp(-pow(v_diff, 2)*150) << std::endl;
			std::cout << dir.transpose() << " " << mHandPosition(1) << " " << mParamCur.transpose() << std::endl;
		}
	}
	return r_param;
}
void
Controller::
UpdateAdaptiveReward()
{

	auto& skel = this->mCharacter->GetSkeleton();
	
	std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), mTargetPositions,
								 skel->getVelocities(), mTargetVelocities, mRewardBodies, false);
	double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();	
	double time_diff = (mAdaptiveStep + 1) - mReferenceManager->GetTimeStep(mPrevFrameOnPhase, true);
	double r_time = exp(-pow(time_diff, 2)*75);

	double r_tracking = 0.8 * accum_bvh + 0.2 * r_time;
	double r_similarity = this->GetSimilarityReward();
	double r_param = this->GetParamReward();

	double r_tot = r_tracking;

	mRewardParts.clear();
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(10 * r_param);
		mRewardParts.push_back(accum_bvh);
		mRewardParts.push_back(r_time);
		mRewardParts.push_back(r_similarity);
	}
	if(r_param != 0) {
		if(mParamRewardTrajectory == 0) {
			mParamRewardTrajectory = r_param;
		}
		else {
			mParamRewardTrajectory *= r_param;
		}
	}
	mTrackingRewardTrajectory += accum_bvh;
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

	double r_time = exp(-pow(mActions[mInterestedDof],2)*40);
	mRewardParts.clear();
	double r_tot = 0.9 * (0.5 * tracking_rewards_bvh[0] + 0.1 * tracking_rewards_bvh[1] + 0.3 * tracking_rewards_bvh[2] + 0.1 * tracking_rewards_bvh[3] ) + 0.1 * r_time;
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
	if(!mRecord && root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 2;
	}
	if(!mRecord && root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
		mIsTerminal = true;
		terminationReason = 1;
	}
	else if(!mRecord && std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 5;
	}
	else if(!mRecord && mCurrentFrame > mReferenceManager->GetPhaseLength()* 3 + 10) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}
	else if(mRecord && mCurrentFrame > mReferenceManager->GetPhaseLength()* 3 + 10) { // this->mBVH->GetMaxFrame() - 1.0){
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
SetGoalParameters(Eigen::VectorXd tp)
{
	mParamGoal = tp;
	// this->mWorld->setGravity(mParamGoal(0)*mBaseGravity);
	// this->SetSkeletonWeight(mParamGoal(1)*mBaseMass);
}

void
Controller::
SetSkeletonWeight(double mass)
{

	double m_new = mass / mMass;

	std::vector<std::tuple<std::string, Eigen::Vector3d, double>> deform;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < n_bnodes; i++){
		std::string name = mCharacter->GetSkeleton()->getBodyNode(i)->getName();
		deform.push_back(std::make_tuple(name, Eigen::Vector3d(1, 1, 1), m_new));
	}
	DPhy::SkeletonBuilder::DeformSkeleton(mCharacter->GetSkeleton(), deform);
	mMass = mCharacter->GetSkeleton()->getMass();
}
void 
Controller::
Reset(bool RSI)
{
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
		this->mParamRewardTrajectory = 0;
		this->mTrackingRewardTrajectory = 0;
		this->mTWRewardTrajectory = 0;
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
	
	if(isAdaptive) {
		Eigen::VectorXd p_obj(mObject->GetSkeleton()->getNumDofs());
		p_obj.setZero();
		p_obj.segment<3>(3) = Eigen::Vector3d(-2.0, 0.0, -2.0);
		mObject->GetSkeleton()->setPositions(p_obj);
		mObject->GetSkeleton()->setVelocities(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->setAccelerations(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->computeForwardKinematics(true,true,true);
	}

	ClearRecord();
	SaveStepInfo();
	mHeadRoot = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	mRootZero = mHeadRoot;
	mCountHead += 1;
	
	mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;
	
	mPrevFrame = mCurrentFrame;
	mPrevFrame2 = mPrevFrame;
	
	while(!mPosQueue.empty())
		mPosQueue.pop();
	while(!mTimeQueue.empty())
		mTimeQueue.pop();
	mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	mTimeQueue.push(0);


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

	ret.resize((num_ee)*12+15);
//	ret.resize((num_ee)*9+12);

	for(int i=0;i<num_ee;i++)
	{		
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		//Eigen::Quaterniond q(transform.linear());
		Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
		ret.segment<9>(9*i) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
							   transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2), 
							   transform.translation();
//		ret.segment<6>(6*i) << rot, transform.translation();
	}


	for(int i=0;i<num_ee;i++)
	{
	    int idx = skel->getBodyNode(mEndEffectors[i])->getParentJoint()->getIndexInSkeleton(0);
		ret.segment<3>(9*num_ee + 3*i) << vel.segment<3>(idx);
//	    ret.segment<3>(6*num_ee + 3*i) << vel.segment<3>(idx);

	}

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	//Eigen::Quaterniond q(transform.linear());

	Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
	Eigen::Vector3d root_angular_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getAngularVelocity();
	Eigen::Vector3d root_linear_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getCOMLinearVelocity();

	ret.tail<15>() << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
					  transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2),
					  transform.translation(), root_angular_vel_relative, root_linear_vel_relative;
//	ret.tail<12>() << rot, transform.translation(), root_angular_vel_relative, root_linear_vel_relative;

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
	// p.resize(p_save.rows()-6);
	// p = p_save.tail(p_save.rows()-6);

	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	int num_p = (n_bnodes - 1) * 6;
	p.resize(num_p);

	for(int i = 1; i < n_bnodes; i++){
		Eigen::Isometry3d transform = skel->getBodyNode(i)->getRelativeTransform();
		// Eigen::Quaterniond q(transform.linear());
		p.segment<6>(6*(i-1)) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
								 transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2);
	}

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
	if(isParametric) {
		state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+1+mParamGoal.rows());
		state<< p, v, up_vec_angle, root_height, p_next, ee, mCurrentFrameOnPhase, mParamGoal;
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
