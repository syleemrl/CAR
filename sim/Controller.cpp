#include "Controller.h"
#include "Character.h"
#include <boost/filesystem.hpp>
#include <Eigen/QR>
#include <fstream>
#include <algorithm>
namespace DPhy
{	

Controller::Controller(std::string motion, bool record)
	:mTimeElapsed(0),mControlHz(30),mSimulationHz(600),mCurrentFrame(0),
	w_p(0.3),w_v(0.3),w_ee(0.2),w_com(0.2), w_srl(0.0),
//	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25), w_srl(0.0),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{
	this->mDeformParameter = std::make_tuple(1.5, 3.0, 1.0);

	this->mRecord = record;
	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();
	this->mWorld->setGravity(Eigen::Vector3d(0,-1.81,0));

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml"));
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);
	
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->SetReference(motion);

	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());

	Eigen::VectorXd kp(this->mCharacter->GetSkeleton()->getNumDofs()), kv(this->mCharacter->GetSkeleton()->getNumDofs());
	
	kp.setZero();
	kv.setZero();
	this->mCharacter->SetPDParameters(kp,kv);

	mInterestedBodies.clear();
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

	mRewardBodies.clear();
	mRewardBodies.push_back("Torso");
	
	mRewardBodies.push_back("FemurR");
	mRewardBodies.push_back("TibiaR");
	mRewardBodies.push_back("FootR");

	mRewardBodies.push_back("FemurL");
	mRewardBodies.push_back("TibiaL");
	mRewardBodies.push_back("FootL");

	mRewardBodies.push_back("Spine");
	mRewardBodies.push_back("Neck");
	mRewardBodies.push_back("Head");

	mRewardBodies.push_back("ForeArmL");
	mRewardBodies.push_back("ArmL");
	mRewardBodies.push_back("HandL");

	mRewardBodies.push_back("ForeArmR");
	mRewardBodies.push_back("ArmR");
	mRewardBodies.push_back("HandR");

	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	this->mCGL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootL"));
	this->mCGR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootR"));
	this->mCGEL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootEndL"));
	this->mCGER = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootEndR"));
	this->mCGHL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("HandL"));
	this->mCGHR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("HandR"));
	this->mCGG = collisionEngine->createCollisionGroup(this->mGround.get());

	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()* 3 + 1);
	mActions.setZero();

	mEndEffectors.clear();
	mEndEffectors.push_back("FootR");
	mEndEffectors.push_back("FootL");
	mEndEffectors.push_back("HandL");
	mEndEffectors.push_back("HandR");
	mEndEffectors.push_back("Head");

	mGRFJoints.clear();
	mGRFJoints.push_back("FootR");
	mGRFJoints.push_back("FootL");
	mGRFJoints.push_back("FootEndR");
	mGRFJoints.push_back("FootEndL");

	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	//temp
	this->mTargetContacts = Eigen::VectorXd::Zero(6);
	this->mRewardParts.resize(6, 0.0);

	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();
	
	this->torques.clear();
	this->GRFs.clear();
	this->mRecordVelocity.clear();
	this->mRecordPosition.clear();
	this->mRecordTime.clear();
	this->mRecordTimeDT.clear();

}
void 
Controller::
SetReference(std::string motion) 
{
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
	this->mRefCharacter = new DPhy::Character(path);
	this->mRefCharacter->LoadBVHMap(path);

	this->mBVH = new BVH();
	path = std::string(CAR_DIR) + std::string("/motion/") + motion + std::string(".bvh");
	this->mBVH->Parse(path);
	this->mRefCharacter->ReadFramesFromBVH(this->mBVH);
	for(int i = 0; i <= mBVH->GetMaxFrame() * 2; i++) {
		mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, i, true);
	}
	// this->DeformCharacter(std::get<0>(mDeformParameter), std::get<1>(mDeformParameter));
}
const dart::dynamics::SkeletonPtr& 
Controller::GetRefSkeleton() { 
	return this->mRefCharacter->GetSkeleton(); 
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

	// set action target pos
	int num_body_nodes = this->mInterestedBodies.size();
	double action_multiplier = 0.2;

	for(int i = 0; i < num_body_nodes*3; i++){
		mActions[i] = dart::math::clip(mActions[i]*action_multiplier, -0.7*M_PI, 0.7*M_PI);
	}

	mActions[num_body_nodes*3] = dart::math::clip(mActions[num_body_nodes*3]*action_multiplier, -1.0, 1.0);
	int additionalStep = (int) std::floor(mActions[num_body_nodes*3] * 10) * 2;
	// additionalStep = 0;

	this->mCurrentFrame += 1;

	Frame* p_v_target = mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mCurrentFrame, true);
	this->mTargetPositions = p_v_target->position;
	this->mTargetVelocities = p_v_target->velocity * ((double) this->mSimPerCon / (this->mSimPerCon + additionalStep));
	this->mTargetCOMvelocity = p_v_target->COMvelocity * std::get<2>(mDeformParameter); // mInputVelocity.first; // mRefCharacter->GetCOMVelocity() * 0.7;
	mInputVelocity.second += 1;
	delete p_v_target;

	this->mPDTargetPositions = this->mTargetPositions;
	this->mPDTargetVelocities = this->mTargetVelocities;

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		mPDTargetPositions.segment<3>(idx) += mActions.segment<3>(3*i);
	}

	// set pd gain action
	Eigen::VectorXd kp(mCharacter->GetSkeleton()->getNumDofs()), kv(mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		if(mInterestedBodies[i] == "Spine" || mInterestedBodies[i] == "FemurR" || mInterestedBodies[i] == "FemurL"){
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(1000);
		}
		else{
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(500);
		}
	}

	// KV_RATIO from CharacterConfiguration.h
	kv = KV_RATIO * kp;
	mCharacter->SetPDParameters(kp, kv);
	Eigen::VectorXd torque;
	mTorque.setZero();
	for(int i = 0; i < this->mSimPerCon + additionalStep; i += 2){
		torque = mCharacter->GetSPDForces(mPDTargetPositions, mPDTargetVelocities);
		mTorque += torque*2;
		for(int j = 0; j < 2; j++)
		{
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step();
		}
		this->mTimeElapsed += 2.0;
		if(this->mTimeElapsed % (int) mSimPerCon == 0) {
			SaveDisplayInfo();
		}
	}
	mTorque /= (this->mSimPerCon + additionalStep);
	this->UpdateGRF(mGRFJoints);
	this->UpdateReward();
	this->UpdateTerminalInfo();
	// this->UpdateGRF(mGRFJoints);

	if(mRecord) {
		std::cout << additionalStep << std::endl;
		this->torques.push_back(torque);
		this->mRecordTime.push_back(this->mTimeElapsed / (double) this->mSimPerCon);
		this->mRecordTimeDT.push_back(additionalStep);
	}

}
void
Controller::
SaveDisplayInfo() 
{
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	
	bool rightContact = CheckCollisionWithGround("FootEndR") || CheckCollisionWithGround("FootR");
	bool leftContact = CheckCollisionWithGround("FootEndL") || CheckCollisionWithGround("FootL");

	mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));
}
void
Controller::
UpdateReward()
{
	auto& skel = this->mCharacter->GetSkeleton();

	//Position Differences
	Eigen::VectorXd p_diff = skel->getPositionDifferences(this->mTargetPositions, skel->getPositions());

	//Velocity Differences
	Eigen::VectorXd v_diff = skel->getVelocityDifferences(this->mTargetVelocities, skel->getVelocities());

	Eigen::VectorXd p_diff_reward, v_diff_reward;
	int num_reward_body_nodes = this->mRewardBodies.size();

	p_diff_reward.resize(num_reward_body_nodes*3);
	v_diff_reward.resize(num_reward_body_nodes*3);

	for(int i = 0; i < num_reward_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mRewardBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		p_diff_reward.segment<3>(3*i) = p_diff.segment<3>(idx);
		v_diff_reward.segment<3>(3*i) = v_diff.segment<3>(idx);
	}

	//End-effector position and COM Differences
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	std::vector<Eigen::Isometry3d> ee_transforms;
	Eigen::VectorXd ee_diff(mEndEffectors.size()*3);
	Eigen::Vector3d com_diff;

	Eigen::Matrix3d cur_root_ori_inv = skel->getRootBodyNode()->getWorldTransform().linear().inverse();
	Eigen::Vector3d com_v_diff = cur_root_ori_inv * skel->getRootBodyNode()->getCOMLinearVelocity();
	
	for(int i=0;i<mEndEffectors.size();i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
		// ee_transforms.push_back(cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform());

	}
	
	com_diff = skel->getCOM();

	skel->setPositions(mPDTargetPositions);
	skel->setVelocities(mPDTargetVelocities);
	skel->computeForwardKinematics(true,true,false);
	
	for(int i=0;i<mEndEffectors.size();i++){
	//	Eigen::Isometry3d diff = ee_transforms[i].inverse() * target_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee_diff.segment<3>(3*i) = diff.translation();
	}
	Eigen::Matrix3d target_root_ori_inv = skel->getRootBodyNode()->getWorldTransform().linear().inverse();
	com_v_diff -= target_root_ori_inv * this->mTargetCOMvelocity; 
	com_diff -= skel->getCOM();
	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	double scale = 1.0;

	//mul
	double sig_p = 0.1 * scale; 		// 2
	double sig_v = 1.0 * scale;		// 3
	double sig_com = 0.3 * scale;		// 4
	double sig_ee = 0.3 * scale;		// 8
	double sig_com_v = 0.5 * scale;

	//sum
	// double sig_p = 0.15 * scale; 		// 2
	// double sig_v = 1.5 * scale;		// 3
	// double sig_com = 0.09 * scale;		// 4
	// double sig_ee = 0.08 * scale;		// 8
	// double sig_srl = 1.5 * scale;

	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_v = exp_of_squared(v_diff_reward,sig_v);
	double r_ee = exp_of_squared(ee_diff,sig_ee);
	double r_com = exp_of_squared(com_diff,sig_com);
	double r_com_v = exp_of_squared(com_v_diff,sig_com_v);
//	double r_torque = exp_of_squared(2.0, mTorque);
	double r_a = exp(-0.5*fabs(mActions[mInterestedBodies.size()*3]));
	double r_tot = r_p*r_v*r_com*r_ee*r_a;
	
	// double r_tot =  w_p*r_p 
	// 				+ w_v*r_v 
	// 				+ w_com*r_com
	// 				+ w_ee*r_ee;
	mRewardParts.clear();
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(6, 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(r_p);
		mRewardParts.push_back(r_v);
		mRewardParts.push_back(r_com);
		mRewardParts.push_back(r_ee);
		mRewardParts.push_back(r_a);
	}

	// if(mInputVelocity.second >= 40) {
	// 	double dv = dart::math::Random::uniform(-0.2, 0.2);
	// 	if(dv >= -0.1 && dv <= 0.1) {
	// 		mInputVelocity.first += dv;
	// 		if(mInputVelocity.first > 1.5) {
	// 			mInputVelocity.first = 1.5 - abs(dv);
	// 		} else if(mInputVelocity.first < 0.5) {
	// 			mInputVelocity.first = 0.5 + abs(dv);
	// 		}
	// 	}
	// 	mInputVelocity.second = 0;
	// }

}
void
Controller::
UpdateTerminalInfo()
{	
	auto& skel = mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];
	Eigen::Vector3d root_pos_diff = this->mTargetPositions.segment<3>(3) - root_pos;

	skel->setPositions(this->mTargetPositions);
	skel->computeForwardKinematics(true, false, false);
	Eigen::Isometry3d root_diff = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	
	Eigen::AngleAxisd root_diff_aa(root_diff.linear());
	double angle = RadianClamp(root_diff_aa.angle());

	skel->setPositions(p);
	skel->computeForwardKinematics(true, false, false);

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
	else if(this->mCurrentFrame > 1000 ) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}
}
bool
Controller::
FollowBvh()
{	
	if(IsTerminalState())
		return false;
	auto& skel = mCharacter->GetSkeleton();

	Frame* p_v_target = mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mCurrentFrame);
	mTargetPositions = p_v_target->position;
	mTargetVelocities = p_v_target->velocity;
	delete p_v_target;

	for(int i=0;i<this->mSimPerCon;i++)
	{
		skel->setPositions(mTargetPositions);
		skel->setVelocities(mTargetVelocities);
		skel->computeForwardKinematics(true, true, false);
	}
	this->mCurrentFrame += this->mStep;
	this->mTimeElapsed += 1.0 / this->mControlHz;
	return true;
}
void
Controller::
DeformCharacter(double w0,double w1)
{

	std::vector<std::tuple<std::string, Eigen::Vector3d, double>> deform;
	deform.push_back(std::make_tuple("Head", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));

	deform.push_back(std::make_tuple("Torso", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("Spine", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));

	deform.push_back(std::make_tuple("ForeArmL", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("ArmL", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("ForeArmR", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("ArmR", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("HandL", Eigen::Vector3d(w0, 1, w1), w1*1*w0));
	deform.push_back(std::make_tuple("HandR", Eigen::Vector3d(w0, 1, w1), w1*1*w0));

	deform.push_back(std::make_tuple("FemurL", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("TibiaL", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("FemurR", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("TibiaR", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	deform.push_back(std::make_tuple("FootR", Eigen::Vector3d(w1, 1, w0), w1*w1*w0));
	deform.push_back(std::make_tuple("FootEndR", Eigen::Vector3d(w1, 1, w0), w1*w1*w0));
	deform.push_back(std::make_tuple("FootL", Eigen::Vector3d(w1, 1, w0), w1*1*w0));
	deform.push_back(std::make_tuple("FootEndL", Eigen::Vector3d(w1, 1, w0), w1*1*w0));

	DPhy::SkeletonBuilder::DeformSkeleton(mRefCharacter->GetSkeleton(), deform);
	DPhy::SkeletonBuilder::DeformSkeleton(mCharacter->GetSkeleton(), deform);
	

	this->mRefCharacter->RescaleOriginalBVH(std::sqrt(w0));

}
void 
Controller::
Reset(bool RSI)
{
	this->mWorld->reset();
	auto& skel = mCharacter->GetSkeleton();
	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();

	p.setZero();
	v.setZero();
	skel->setPositions(p);
	skel->setVelocities(v);
	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();
	skel->computeForwardKinematics(true,true,false);

	//RSI
	if(RSI) {
		this->mCurrentFrame = mRefCharacter->GetMaxFrame() + (int) dart::math::Random::uniform(0.0, this->mBVH->GetMaxFrame()-5);
		this->mTimeElapsed = this->mCurrentFrame * this->mSimPerCon;
	}
	else {
		this->mTimeElapsed = 0; // 0.0;
		this->mCurrentFrame = 0; // 0;
	}
	this->mStartFrame = this->mCurrentFrame;

	Frame* p_v_target = mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mCurrentFrame, true);
	this->mTargetPositions = p_v_target->position;
	this->mTargetVelocities = p_v_target->velocity;
	delete p_v_target;

	mInputVelocity = std::make_pair(dart::math::Random::uniform(0.5, 1.5), 0);
	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;
	this->mStep = 1;

	this->mRewardParts.resize(6, 0.0);

	this->torques.clear();
	this->GRFs.clear();
	this->mRecordVelocity.clear();
	this->mRecordPosition.clear();
	this->mRecordTime.clear();

	SaveDisplayInfo();

	bool rightContact = CheckCollisionWithGround("FootEndR") || CheckCollisionWithGround("FootR");
	bool leftContact = CheckCollisionWithGround("FootEndL") || CheckCollisionWithGround("FootL");
	if(leftContact && rightContact) {
		mDoubleStanceInfo = std::make_tuple(true, mTimeElapsed, 0);
	} else {
		mDoubleStanceInfo = std::make_tuple(false, 0, 0);
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
	if(bodyName == "FootR"){
		bool isCollide = collisionEngine->collide(this->mCGR.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "FootL"){
		bool isCollide = collisionEngine->collide(this->mCGL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "FootEndR"){
		bool isCollide = collisionEngine->collide(this->mCGER.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "FootEndL"){
		bool isCollide = collisionEngine->collide(this->mCGEL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "HandR"){
		bool isCollide = collisionEngine->collide(this->mCGHR.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "HandL"){
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
	if(mIsTerminal)
		return Eigen::VectorXd::Zero(this->mNumState);
	auto& skel = mCharacter->GetSkeleton();

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

	Frame* p_v_target = mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mCurrentFrame+1, true);
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->position, p_v_target->velocity);
	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	Eigen::VectorXd state;
	double phase = ((int) mCurrentFrame % (int) mBVH->GetMaxFrame()) / mBVH->GetMaxFrame();
	state.resize(p.rows()+v.rows()+1+p_next.rows()+ee.rows());
	state<< p, v, up_vec_angle, p_next, ee; //, mInputVelocity.first;
	// state.resize(p.rows()+v.rows()+1+1+ee.rows());
	// state<< p, v, up_vec_angle, phase, ee; //, mInputVelocity.first;
	return state;
}
void
Controller::SaveHistory(const std::string& filename) {
	std::cout << "save results" << std::endl;
	std::ofstream ofs(filename);
	std::cout << this->torques.size() << std::endl;
	ofs << this->torques.size() << std::endl;
	int i = 0;
	for(auto& t: this->torques) {
		ofs << t.transpose() << std::endl;
	}
	ofs << this->GRFs.at(0).size() << std::endl;
	for(auto& g: this->GRFs) {
		for(int i = 0; i < g.size(); i++) 
		{
			ofs << g.at(i).transpose() << std::endl;
		}
	}
	for(auto t: this->mRecordTime) {
		ofs << t << std::endl;
	}
	for(auto t: this->mRecordTimeDT) {
		ofs << t << std::endl;
	}
	ofs.close();
}
std::vector<Eigen::VectorXd>
Controller::GetGRF() {
	return GRFs.back();
}
void 
Controller::UpdateGRF(std::vector<std::string> joints) {
	std::vector<Eigen::VectorXd> grf;
	auto& skel = mCharacter->GetSkeleton();

	auto contacts =  mWorld->getConstraintSolver()->getLastCollisionResult().getContacts();

	for(int j = 0; j < joints.size(); j++)
	{
		Eigen::Isometry3d cur_inv = skel->getBodyNode(joints.at(j))->getWorldTransform().inverse();
		int idx = 3 * mCharacter->GetSkeleton()->getBodyNode(joints.at(j))->getIndexInSkeleton() + 3;

		Eigen::MatrixXd Jt_com = skel->getLinearJacobian(skel->getBodyNode(joints.at(j)), cur_inv * skel->getBodyNode(joints.at(j))->getCOM()).transpose();
		Eigen::Vector3d t_total(0, 0, 0);

		for(int i = 0; i < contacts.size(); i++) {
			if(contacts.at(i).collisionObject2->getShapeFrame()->getName().find(joints.at(j)) != std::string::npos) {
				t_total += skel->getLinearJacobian(skel->getBodyNode(joints.at(j)), cur_inv * contacts.at(i).point).block<3, 3>(0, idx).transpose() * (-contacts.at(i).force);
			}
		}
		Eigen::MatrixXd Jt_com_inv = Jt_com.completeOrthogonalDecomposition().pseudoInverse();
		Eigen::Vector3d f_total = Jt_com_inv.block<3, 3>(0, idx) * t_total;

		Eigen::Vector6d result;
		result << skel->getBodyNode(joints.at(j))->getCOM(), f_total;
		grf.push_back(result);
	}

	if(mRecord) GRFs.push_back(grf);

	bool rightContact = CheckCollisionWithGround("FootEndR") || CheckCollisionWithGround("FootR");
	bool leftContact = CheckCollisionWithGround("FootEndL") || CheckCollisionWithGround("FootL");
	
	// if(!std::get<0>(mDoubleStanceInfo) && rightContact && leftContact) {
	// 	mDoubleStanceInfo = std::make_tuple(true, mTimeElapsed, std::get<2>(mDoubleStanceInfo));
	// 	std::cout << "double stance start :" << (double)mTimeElapsed / mSimPerCon << std::endl;
	// } else if(std::get<0>(mDoubleStanceInfo) && !rightContact && leftContact) {
	// 	mDoubleStanceInfo= std::make_tuple(false,  mTimeElapsed - std::get<1>(mDoubleStanceInfo), grf[3][4]);
	// 	std::cout << "double stance end: " << (double)mTimeElapsed / mSimPerCon << " " << grf[3].transpose() << " " << grf[3][4] / (mTimeElapsed - std::get<1>(mDoubleStanceInfo)) << std::endl;
	// } else if(std::get<0>(mDoubleStanceInfo) && rightContact && !leftContact) {
	// 	mDoubleStanceInfo= std::make_tuple(false,  mTimeElapsed - std::get<1>(mDoubleStanceInfo), grf[2][4]);
	// 	std::cout << "double stance end: " << (double)mTimeElapsed / mSimPerCon << " " << grf[2].transpose() << " " << grf[2][4] / (mTimeElapsed - std::get<1>(mDoubleStanceInfo)) << std::endl;
	// }

}

std::string 
Controller::GetContactNodeName(int i) { 
	return mCharacter->GetContactNodeName(i); 
}

}
