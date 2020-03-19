#include "Controller.h"
#include "Character.h"
#include <boost/filesystem.hpp>
#include <Eigen/QR>
#include <fstream>
#include <algorithm>
namespace DPhy
{	

Controller::Controller(ReferenceManager* ref, std::string stats, bool adaptive, bool record)
	:mControlHz(30),mSimulationHz(600),mCurrentFrame(0),
	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25), w_srl(0.0),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{
	this->sig_torque = 0.4;
	this->mRescaleParameter = std::make_tuple(1.0, 1.0, 1.0);
	this->isAdaptive = adaptive;
	this->mRecord = record;
	this->mReferenceManager = ref;
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
	// this->RescaleCharacter(1, 1);

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
	this->mRewardParts.resize(7, 0.0);

	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();
	
	this->GRFs.clear();
	this->mRecordVelocity.clear();
	this->mRecordPosition.clear();
	this->mRecordCOM.clear();
	this->mRecordTime.clear();
	this->mRecordDTime.clear();
	this->mRecordDCOM.clear();
	this->mRecordWork.clear();
	this->mRecordTorque.clear();
	this->mRecordWorkByJoints.clear();
	this->mRecordTorqueByJoints.clear();

	this->mMask.resize(dof);
	this->mMask.setZero();
	
	mControlFlag.resize(2);

	int num_body_nodes = this->mInterestedBodies.size();
	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		if(mInterestedBodies[i].find("Tibia") != std::string::npos ||
			mInterestedBodies[i].find("Femur") != std::string::npos ||
			mInterestedBodies[i].find("Spine") != std::string::npos ||
			mInterestedBodies[i].find("Foot") != std::string::npos ) {
			// this->mMask[idx] = 1;
			this->mMask.segment<3>(idx) = Eigen::Vector3d::Constant(1);
		}
	}
	
	mRewardLabels.clear();
	if(isAdaptive) {
		mRewardLabels.push_back("total_d");
		mRewardLabels.push_back("total_s");
		mRewardLabels.push_back("p");
		mRewardLabels.push_back("com");
		mRewardLabels.push_back("h");
	} else {
		mRewardLabels.push_back("total");
		mRewardLabels.push_back("p");
		mRewardLabels.push_back("v");
		mRewardLabels.push_back("com");
		mRewardLabels.push_back("ee");
		mRewardLabels.push_back("t");
	}
}
void
Controller::
UpdateSigTorque()
{
	// if(sig_torque < 1) sig_torque *= 2;
	std::cout << "sig torque updated : " << sig_torque << std::endl;
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

	for(int i = 0; i < num_body_nodes*3; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.2, -0.7*M_PI, 0.7*M_PI);
	}

	mActions[num_body_nodes*3] = dart::math::clip(mActions[num_body_nodes*3]*0.4, -0.8, 0.8);
	mAdaptiveStep = mActions[num_body_nodes*3];
	this->mCurrentFrame += (1 + mAdaptiveStep);
	this->mCurrentFrameOnPhase += (1 + mAdaptiveStep);
	if(this->mCurrentFrameOnPhase > mReferenceManager->GetPhaseLength() - 1){
		this->mCurrentFrameOnPhase -= (mReferenceManager->GetPhaseLength() - 1);
		mControlFlag.setZero();
	}

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity() * (1 + mAdaptiveStep);

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

	// double work_sum = 0;
	// Eigen::VectorXd torque_sum(this->mCharacter->GetSkeleton()->getNumDofs());
	// Eigen::VectorXd work_sum_joints(this->mCharacter->GetSkeleton()->getNumDofs() / 3 - 2);
	// Eigen::VectorXd torque_sum_joints(this->mCharacter->GetSkeleton()->getNumDofs() / 3 - 2);
	// torque_sum.setZero();
	// torque_sum_joints.setZero();
	// work_sum_joints.setZero();

	for(int i = 0; i < this->mSimPerCon; i += 2){
		torque = mCharacter->GetSPDForces(mPDTargetPositions, mPDTargetVelocities);
		Eigen::VectorXd torque_masked = torque.cwiseProduct(this->mMask);

		for(int j = 0; j < 2; j++)
		{
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step();

			// Eigen::VectorXd curVelocity = mCharacter->GetSkeleton()->getVelocities();
			// work_sum += torque.dot(curVelocity) * 1.0 / mSimulationHz;
			// torque_sum += torque * 1.0 / mSimulationHz;
			// if(mRecord) 
			// {
			// 	for(int i = 6; i < curVelocity.rows(); i += 3) {
			// 		work_sum_joints[i/3 - 2] += torque.segment<3>(i).dot(curVelocity.segment<3>(i)) * 1.0 / mSimulationHz;
			// 		torque_sum_joints[i/3 - 2] += torque.segment<3>(i).norm() * 1.0 / mSimulationHz;
			// 	}

			// }
			// // work_sum += torque_masked.cwiseAbs().dot(curVelocity.cwiseAbs()) * 1.0 / mSimulationHz;
			// auto skel = this->mCharacter->GetSkeleton();
		}
		mTimeElapsed += 2 * (1 + mAdaptiveStep);
	}
	if(mRecord) {
		SaveStepInfo();
	}
	nTotalSteps += 1;
	// mRecordWork.push_back(work_sum);
	// mRecordTorque.push_back(torque_sum / this->mSimPerCon);
	// if(mRecord) {
	// 	this->mRecordTorqueByJoints.push_back(torque_sum_joints);
	// 	mRecordWorkByJoints.push_back(work_sum_joints);
	// }
	if(isAdaptive)
		this->UpdateAdaptiveReward();
	else
		this->UpdateReward();
	this->UpdateTerminalInfo();
	mPrevPositions = mCharacter->GetSkeleton()->getPositions();

}
void
Controller::
SaveStepInfo() 
{
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	mRecordCOM.push_back(mCharacter->GetSkeleton()->getCOM());
	mRecordTime.push_back(mCurrentFrame);
	bool rightContact = CheckCollisionWithGround("FootEndR") || CheckCollisionWithGround("FootR");
	bool leftContact = CheckCollisionWithGround("FootEndL") || CheckCollisionWithGround("FootL");

	mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));
}
void
Controller::
UpdateAdaptiveReward()
{
	auto& skel = this->mCharacter->GetSkeleton();

	//Position Differences
	Eigen::VectorXd p_diff = skel->getPositionDifferences(this->mTargetPositions, skel->getPositions());

	Eigen::VectorXd p_diff_reward;
	int num_reward_body_nodes = this->mRewardBodies.size();

	p_diff_reward.resize(num_reward_body_nodes*3);
	for(int i = 0; i < num_reward_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mRewardBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		p_diff_reward.segment<3>(3*i) = p_diff.segment<3>(idx);
	}

	//End-effector position and COM Differences
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	Eigen::Vector3d com_diff;

	Eigen::Matrix3d cur_root_ori_inv = skel->getRootBodyNode()->getWorldTransform().linear().inverse();
	
	com_diff = skel->getCOM();

	skel->setPositions(this->mTargetPositions);
	skel->setVelocities(this->mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	com_diff -= skel->getCOM();
	com_diff[1] = 0;
	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	double scale = 1.0;

	double sig_p = 0.1 * scale; 		// 2
	double sig_com = 0.1 * scale;		// 4

	double r_height = 0;
	if(mCurrentFrameOnPhase >= 44.5 && mControlFlag[0] == 0) {
		double height_diff = skel->getCOM()[1] - 1.15;
		r_height = exp(-pow(height_diff, 2) * 10)*1.5;
		mControlFlag[0] = 1;
	//	std::cout << height_diff << " " << r_height << std::endl;
	}
	// else if(mCurrentFrameOnPhase >= 55 && mControlFlag[1] == 0) {
	// 	Eigen::VectorXd height_diff(4);
	// 	height_diff[0] = skel->getBodyNode("FootL")->getWorldTransform().translation()[1] - 0.03;
	// 	height_diff[1] = skel->getBodyNode("FootR")->getWorldTransform().translation()[1] - 0.03;
	// 	height_diff[2] = skel->getBodyNode("FootEndL")->getWorldTransform().translation()[1] - 0.03;
	// 	height_diff[3] = skel->getBodyNode("FootEndR")->getWorldTransform().translation()[1] - 0.03;
	// 	for(int i = 0; i < height_diff.rows(); i++) {
	// 		if(height_diff[i] < 0) height_diff[i] = 0;
	// 	}
	// 	r_height = exp_of_squared(height_diff, 0.2)*0.75;
	// 	mControlFlag[1] = 1;
	// //	std::cout << height_diff.transpose() << " " << r_height << std::endl;
	// }
	// std::cout << nTotalSteps << " : " << mCurrentFrameOnPhase << std::endl;
	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_com = exp_of_squared(com_diff,sig_com);

	double r_tot_dense = r_com;
	mRewardParts.clear();
	if(dart::math::isNan(r_p)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot_dense);
		mRewardParts.push_back(r_height*3);
		mRewardParts.push_back(r_p);
		mRewardParts.push_back(r_com);
		mRewardParts.push_back(r_height);
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
	ee_diff.setZero();
	Eigen::Vector3d com_diff;

	Eigen::Matrix3d cur_root_ori_inv = skel->getRootBodyNode()->getWorldTransform().linear().inverse();
	
	std::vector<bool> isContact;
	for(int i=0;i<mEndEffectors.size();i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
	//	isContact.push_back(CheckCollisionWithGround(mEndEffectors[i]));
	}
	
	com_diff = skel->getCOM();

	skel->setPositions(this->mTargetPositions);
	skel->setVelocities(this->mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	for(int i=0;i<mEndEffectors.size();i++){
	//	if(isContact[i]) {
			Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
			ee_diff.segment<3>(3*i) = diff.translation();
	//	}
	}
	com_diff -= skel->getCOM();

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

	double scale = 1.0;

	double sig_p = 0.1 * scale; 		// 2
	double sig_v = 1.0 * scale;		// 3
	double sig_com = 0.1 * scale;		// 4
	double sig_ee = 0.2 * scale;		// 8
	double w_a = 0.1;

	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_v = exp_of_squared(v_diff_reward,sig_v);
	double r_ee = exp_of_squared(ee_diff,sig_ee);
	double r_com = exp_of_squared(com_diff,sig_com);
	double r_a = exp(-pow(mAdaptiveStep, 2)*100);
	double r_tot = w_p*r_p + w_v*r_v + w_com*r_com + w_ee*r_ee;
	r_tot = 0.9 * r_tot + 0.1 * r_a;
	mRewardParts.clear();
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(r_p);
		mRewardParts.push_back(r_v);
		mRewardParts.push_back(r_com);
		mRewardParts.push_back(r_ee);
		mRewardParts.push_back(r_a);
	}

}
void
Controller::
UpdateTerminalInfo()
{	
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	Eigen::VectorXd targetPositions = p_v_target->GetPosition();
	Eigen::VectorXd targetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(targetPositions);
	skel->setVelocities(targetVelocities);
	skel->computeForwardKinematics(true,true,false);

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
	else if(this->nTotalSteps > 1000 ) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}

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
RescaleCharacter(double w0,double w1)
{

	std::vector<std::tuple<std::string, Eigen::Vector3d, double>> deform;
	// deform.push_back(std::make_tuple("Head", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));

	// deform.push_back(std::make_tuple("Torso", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("Spine", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));

	// deform.push_back(std::make_tuple("ForeArmL", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("ArmL", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("ForeArmR", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("ArmR", Eigen::Vector3d(w0, w1, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("HandL", Eigen::Vector3d(w0, 1, w1), w1*1*w0));
	// deform.push_back(std::make_tuple("HandR", Eigen::Vector3d(w0, 1, w1), w1*1*w0));

	// deform.push_back(std::make_tuple("FemurL", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("TibiaL", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("FemurR", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("TibiaR", Eigen::Vector3d(w1, w0, w1), w1*w1*w0));
	// deform.push_back(std::make_tuple("FootR", Eigen::Vector3d(w1, 1, w0), w1*w1*w0));
	// deform.push_back(std::make_tuple("FootEndR", Eigen::Vector3d(w1, 1, w0), w1*w1*w0));
	// deform.push_back(std::make_tuple("FootL", Eigen::Vector3d(w1, 1, w0), w1*1*w0));
	// deform.push_back(std::make_tuple("FootEndL", Eigen::Vector3d(w1, 1, w0), w1*1*w0));

	// DPhy::SkeletonBuilder::DeformSkeleton(mCharacter->GetSkeleton(), deform);
	
	// if(w0 != 1) mReferenceManager->RescaleMotion(std::sqrt(w0));


	// deform.push_back(std::make_tuple("ForeArmR", Eigen::Vector3d(1, 1, 1), 4));
	// deform.push_back(std::make_tuple("ArmR", Eigen::Vector3d(1, 1, 1), 4));
	// deform.push_back(std::make_tuple("HandR", Eigen::Vector3d(1, 1, 1), 4));

	// deform.push_back(std::make_tuple("ForeArmL", Eigen::Vector3d(1, 1, 1), 4));
	// deform.push_back(std::make_tuple("ArmL", Eigen::Vector3d(1, 1, 1), 4));
	// deform.push_back(std::make_tuple("HandL", Eigen::Vector3d(1, 1, 1), 4));


	deform.push_back(std::make_tuple("FemurR", Eigen::Vector3d(1, 1, 1), 4));
	deform.push_back(std::make_tuple("TibiaR", Eigen::Vector3d(1, 1, 1), 4));
	deform.push_back(std::make_tuple("FootR", Eigen::Vector3d(1, 1, 1), 4));
	deform.push_back(std::make_tuple("FootEndR", Eigen::Vector3d(1, 1, 1), 4));

	deform.push_back(std::make_tuple("FemurL", Eigen::Vector3d(1, 1, 1), 4));
	deform.push_back(std::make_tuple("TibiaL", Eigen::Vector3d(1, 1, 1), 4));
	deform.push_back(std::make_tuple("FootL", Eigen::Vector3d(1, 1, 1), 4));
	deform.push_back(std::make_tuple("FootEndL", Eigen::Vector3d(1, 1, 1), 4));
	DPhy::SkeletonBuilder::DeformSkeleton(mCharacter->GetSkeleton(), deform);
	std::cout << "Deform done "  << std::endl;

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
	if(RSI && !isAdaptive) {
		this->mCurrentFrame = (int) dart::math::Random::uniform(0.0, mReferenceManager->GetPhaseLength()-5.0);
	}
	else {
		this->mCurrentFrame = 0; // 0;
	}
	this->mCurrentFrameOnPhase =  this->mCurrentFrame;
	this->mStartFrame = this->mCurrentFrame;
	this->nTotalSteps = 0;
	this->mTimeElapsed = 0;
	this->mControlFlag.setZero();

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;

	this->mRewardParts.resize(7, 0.0);

	this->GRFs.clear();
	this->mRecordVelocity.clear();
	this->mRecordPosition.clear();
	this->mRecordCOM.clear();
	this->mRecordTime.clear();
	this->mRecordDTime.clear();
	this->mRecordDCOM.clear();
	this->mRecordEnergy.clear();
	this->mRecordWork.clear();
	this->mRecordTorque.clear();
	this->mRecordWorkByJoints.clear();
	this->mRecordTorqueByJoints.clear();
	this->mRecordFootConstraint.clear();

	SaveStepInfo();

	int num_body_nodes = this->mInterestedBodies.size();

	double energy = 0;
	for(int i = 0; i < num_body_nodes; i++) {
		energy += (mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getPotentialEnergy(Eigen::Vector3d(0,-9.81,0))  
			+ mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getKineticEnergy());
	}
	mRecordEnergy.push_back(energy);
	this->mRecordTime.push_back(0);
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

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame+1);
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->GetPosition(), p_v_target->GetVelocity());
	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	Eigen::VectorXd state;
	double phase = ((int) mCurrentFrame % mReferenceManager->GetPhaseLength()) / (double) mReferenceManager->GetPhaseLength();
	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+1);
	state<< p, v, up_vec_angle, root_height, p_next, ee, mCurrentFrameOnPhase; //, mInputVelocity.first;
	// state.resize(p.rows()+v.rows()+1+1+ee.rows());
	// state<< p, v, up_vec_angle, phase, ee; //, mInputVelocity.first;
	return state;
}
void
Controller::SaveDisplayedData(std::string directory) {
	std::string path = std::string(CAR_DIR) + std::string("/") +  directory;
	std::cout << "save results to" << path << std::endl;

	std::ofstream ofs(path);

	ofs << mRecordPosition.size() << std::endl;
	for(auto t: mRecordPosition) {
		ofs << t.transpose() << std::endl;
	}
	std::cout << "saved position: " << mRecordPosition.size() << ", "<< mReferenceManager->GetPhaseLength() << ", " << mRecordPosition[0].rows() << std::endl;

}
void
Controller::SaveStats(std::string directory) {
	std::string path = std::string(CAR_DIR) + std::string("/") +  directory;
	std::cout << "save results to" << path << std::endl;

	std::ofstream ofs(path);

	ofs << mRecordWork.size() << std::endl;
	for(auto t: mRecordWork) {
		ofs << t << std::endl;
	}
	std::cout << "saved work: " << mRecordWork.size() << std::endl;
	
	ofs << mRecordWorkByJoints.size() << std::endl;
	for(auto t: mRecordWorkByJoints) {
		ofs << t.transpose() << std::endl;
	}
	std::cout << "saved work by joints: " << mRecordWorkByJoints.size() << std::endl;

	ofs << mRecordTorqueByJoints.size() << std::endl;
	for(auto t: mRecordTorqueByJoints) {
		ofs << t.transpose() << std::endl;
	}
	std::cout << "saved torque by joints: " << mRecordTorqueByJoints.size() << std::endl;

	ofs.close();

}
std::vector<Eigen::VectorXd>
Controller::GetGRF() {
	return GRFs.back();
}
}
