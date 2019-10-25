#include "Controller.h"
#include "Character.h"
#include <boost/filesystem.hpp>
#include <Eigen/QR>
#include <fstream>
#include <algorithm>
namespace DPhy
{	

Controller::Controller(std::string motion, bool record)
	:mTimeElapsed(0.0),mControlHz(30),mSimulationHz(600),mCurrentFrame(0),
	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25), w_srl(0.05),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{
	this->mRecord = record;
	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();
	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81,0));

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml"));
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);
	
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
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

	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()* 3 + 3);
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
	this->SetReference(motion);

	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	//temp
	this->mTargetContacts = Eigen::VectorXd::Zero(6);
	this->mRewardParts.resize(6, 0.0);

	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();
	
	this->torques.clear();
	this->GRFs.clear();

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

	this->DeformCharacter(1, 5);
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

	for(int i = num_body_nodes*3; i < num_body_nodes*3 + 3; i++){
		mActions[i] = dart::math::clip(mActions[i]*action_multiplier, -1.0, 1.0);
	}
	
	this->mCurrentFrame += 1;

	Frame* p_v_target = mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mCurrentFrame);
	this->mTargetPositions = p_v_target->position;
	this->mTargetVelocities = p_v_target->velocity;
	this->mTargetContacts = p_v_target->contact;
	this->mTargetCOMvelocity = p_v_target->COMvelocity;
	delete p_v_target;

	this->mPDTargetPositions = this->mTargetPositions;
	this->mPDTargetVelocities = this->mTargetVelocities;

	//SRL
	this->mAdaptiveCOMvelocity = this->mTargetCOMvelocity + mActions.segment<3>(num_body_nodes * 3);

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
	for(int i = 0; i < this->mSimPerCon; i += 2){
		torque = mCharacter->GetSPDForces(mPDTargetPositions, mPDTargetVelocities);
		for(int j = 0; j < 2; j++)
		{
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step();
		}
	}

	this->mTimeElapsed += 1.0 / this->mControlHz;

	this->UpdateReward();
	this->UpdateTerminalInfo();
	if(mRecord) {
		UpdateGRF(mGRFJoints);
		this->torques.push_back(torque);

	}

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

	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	Eigen::Vector3d root_ori_diff = p_diff.segment<3>(0);
	Eigen::Vector3d com_lv_diff = cur_root_inv * skel->getRootBodyNode()->getCOMLinearVelocity();
	
	for(int i=0;i<mEndEffectors.size();i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
		// ee_transforms.push_back(cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform());

	}
	
	com_diff = skel->getCOM();

	skel->setPositions(mPDTargetPositions);
	skel->setVelocities(mPDTargetVelocities);
	skel->computeForwardKinematics(true,true,false);
	
	Eigen::Isometry3d target_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	for(int i=0;i<mEndEffectors.size();i++){
	//	Eigen::Isometry3d diff = ee_transforms[i].inverse() * target_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee_diff.segment<3>(3*i) = diff.translation();
	}

	root_ori_diff = p_diff.segment<3>(0);
	com_lv_diff -= target_root_inv * this->mAdaptiveCOMvelocity; 
	Eigen::Vector3d srl_diff = this->mAdaptiveCOMvelocity - skel->getRootBodyNode()->getCOMLinearVelocity();

	com_diff -= skel->getCOM();

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	double scale = 1.0;

	//mul
	// double sig_p = 0.1 * scale; 		// 2
	// double sig_v = 1.0 * scale;		// 3
	// double sig_com = 0.3 * scale;		// 4
	// double sig_ee = 0.3 * scale;		// 8

	//sum
	double sig_p = 0.15 * scale; 		// 2
	double sig_v = 1.5 * scale;		// 3
	double sig_com = 0.09 * scale;		// 4
	double sig_ee = 0.08 * scale;		// 8
	double sig_srl = 1.5 * scale;

	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_v = exp_of_squared(v_diff_reward,sig_v);
	double r_ee = exp_of_squared(ee_diff,sig_ee);
	double r_com = exp_of_squared(com_diff,sig_com);

	double r_srl = exp_of_squared(srl_diff, sig_srl);
	// double r_tot = r_p*r_v*r_com*r_ee;
	double r_tot =  w_p*r_p 
					+ w_v*r_v 
					+ w_com*r_com;
	//				+ w_ee*r_ee;
	//				+ w_srl*r_srl;
	// r_tot = 0.9*r_tot + 0.1*r_contact;
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
		mRewardParts.push_back(r_srl);
	}
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
	if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
		mIsTerminal = true;
		terminationReason = 1;
	}
	else if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 5;
	}
	else if(this->mCurrentFrame > this->mBVH->GetMaxFrame() - 1.0){
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

	std::vector<std::tuple<std::string, int, double>> deform;
	deform.push_back(std::make_tuple("ForeArmL", 0, w0));
	deform.push_back(std::make_tuple("ArmL", 0, w0));
	deform.push_back(std::make_tuple("ForeArmR", 0, w0));
	deform.push_back(std::make_tuple("ArmR", 0, w0));
	deform.push_back(std::make_tuple("FemurL", 1, w0));
	deform.push_back(std::make_tuple("TibiaL", 1, w0));
	deform.push_back(std::make_tuple("FemurR", 1, w0));
	deform.push_back(std::make_tuple("TibiaR", 1, w0));
//	deform.push_back(std::make_tuple("FootL", 2, w));
//	deform.push_back(std::make_tuple("FootR", 2, w));

	DPhy::SkeletonBuilder::DeformSkeletonLength(mRefCharacter->GetSkeleton(), deform);
	DPhy::SkeletonBuilder::DeformSkeletonLength(mCharacter->GetSkeleton(), deform);
	
	std::vector<std::tuple<std::string, double>> deform_m;
	deform_m.push_back(std::make_tuple("ForeArmL", 0.5 * w1));
	deform_m.push_back(std::make_tuple("ArmL", 0.5 * w1));
	deform_m.push_back(std::make_tuple("ForeArmR", 0.5 * w1));
	deform_m.push_back(std::make_tuple("ArmR", 0.5 * w1));
	deform_m.push_back(std::make_tuple("FemurL", 1.25 * w1));
	deform_m.push_back(std::make_tuple("TibiaL", w1));
	deform_m.push_back(std::make_tuple("FemurR", 1.25 * w1));
	deform_m.push_back(std::make_tuple("TibiaR", w1));
	deform_m.push_back(std::make_tuple("Torso", 1.25 * w1));
	deform_m.push_back(std::make_tuple("Spine", 1.25 * w1));
	deform_m.push_back(std::make_tuple("Neck", 0.5 * w1));
	deform_m.push_back(std::make_tuple("Head", 0.5 * w1));

	DPhy::SkeletonBuilder::DeformSkeletonMass(mRefCharacter->GetSkeleton(), deform_m);
	DPhy::SkeletonBuilder::DeformSkeletonMass(mCharacter->GetSkeleton(), deform_m);

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
		this->mTimeElapsed =  dart::math::Random::uniform(0.0, this->mBVH->GetMaxTime() - 10 /this->mControlHz);
		this->mCurrentFrame = std::floor(this->mTimeElapsed*this->mControlHz);
	}
	else {
		this->mTimeElapsed = 0.0 / this->mControlHz; // 0.0;
		this->mCurrentFrame = 0; // 0;
	}
	this->mStartFrame = this->mCurrentFrame;

	Frame* p_v_target = mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mCurrentFrame);
	this->mTargetPositions = p_v_target->position;
	this->mTargetVelocities = p_v_target->velocity;
	delete p_v_target;

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;
	this->mStep = 1;

	this->mRewardParts.resize(6, 0.0);

	this->torques.clear();
	this->GRFs.clear();

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

	Frame* p_v_target = mRefCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mCurrentFrame+1);
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->position, p_v_target->velocity);
	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	
	Eigen::VectorXd state;
	// state.resize(p.rows()+1+v.rows()+p_next.rows()+ee.rows());
	// state<<p, up_vec_angle, v, p_next, ee;
	state.resize(p.rows()+v.rows()+p_next.rows()+ee.rows());
	state<< p, v, p_next, ee;
	return state;
}
void
Controller::SaveHistory(const std::string& filename) {
	std::cout << "save results" << std::endl;
	std::ofstream ofs(filename);
	std::cout << this->torques.size() << " " << this->GRFs.size() << std::endl;
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

	GRFs.push_back(grf);
}

std::string 
Controller::GetContactNodeName(int i) { 
	return mCharacter->GetContactNodeName(i); 
}

}
