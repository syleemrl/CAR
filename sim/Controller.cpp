#include "Controller.h"
#include "Character.h"
#include <boost/filesystem.hpp>
#include <fstream>

namespace DPhy
{
Controller::Controller()
	:mTimeElapsed(0.0),mControlHz(30),mSimulationHz(600),mActions(Eigen::VectorXd::Zero(0)),
	,w_p(0.2),w_v(0.1),w_ee(0.2),w_com(0.25),w_root_ori(0.1),w_root_av(0.05),
	,terminationReason(-1),mIsNanAtTerminal(false)
{
	this->mWorld = std::make_shared<dart::simulation::World>();
	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81,0));

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(DPHY_DIR)+std::string("/character/ground.xml"));
	this->mGround->GetSkeleton()->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround->GetSkeleton());

	this->mCharacter = new Character::LoadBVHMap(std::string(DPHY_DIR)+std::string("/character/humanoid_new.xml"));
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());

	this->mPositionUpperLimits = this->mCharacter->GetSkeleton()->getPositionUpperLimits();
	this->mPositionLowerLimits = this->mCharacter->GetSkeleton()->getPositionLowerLimits();
	this->mActionRange = (this->mPositionUpperLimits - this->mPositionLowerLimits)*0.1;

	this->mMaxJoint = Eigen::VectorXd::Constant(this->mCharacter->GetSkeleton()->getNumDofs(), -20.);
	this->mMinJoint = Eigen::VectorXd::Constant(this->mCharacter->GetSkeleton()->getNumDofs(), 20.);

	Eigen::VectorXd kp(this->mCharacter->GetSkeleton()->getNumDofs()), kv(this->mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();

	kv = kp * 0.1;
	this->mCharacter->SetPDParameters(kp,kv);

	mInterestedBodies.clear();
	mInterestedBodies.push_back("Spine");
	mInterestedBodies.push_back("Neck");
	mInterestedBodies.push_back("Head");

	mInterestedBodies.push_back("ForeArmL");
	mInterestedBodies.push_back("ArmL");
	mInterestedBodies.push_back("HandL");

	mInterestedBodies.push_back("ForeArmR");
	mInterestedBodies.push_back("ArmR");
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
	this->mCGG = collisionEngine->createCollisionGroup(this->mGround->GetSkeleton().get());

	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()*3);
	mActions.setZero();

	mEndEffectors.clear();
	mEndEffectors.push_back("FootR");
	mEndEffectors.push_back("FootL");
	mEndEffectors.push_back("HandL");
	mEndEffectors.push_back("HandR");
	mEndEffectors.push_back("Head");

	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 
	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	this->mRootCOMAtTerminal.setZero();
	this->mRootCOMAtTerminalRef.setZero();

	this->mIsTerminal = false;
	this->mIsNanAtTerminal = false;
	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();
	torques.clear();
}
void Controller::SetReference(std::string file) 
{
	this->mBVH = new BVH();
	this->mBVH->Parse(file);
	this->mCharacter->InitializeBVH(this->mBVH);
}
void Controller::Step()
{
	int per = mSimulationHz/mControlHz;
	if(IsTerminalState())
		return;
	
	// set action target pos
	int num_body_nodes = this->mInterestedBodies.size();
	double pd_gain_offset = 0;

	auto p_v_target = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mTimeElapsed);
	this->mTargetPositions = std::get<0>(target_tuple);
	this->mTargetVelocities = std::get<1>(target_tuple);
	this->mModifiedTargetPositions = this->mTargetPositions;
	this->mModifiedTargetVelocities = this->mTargetVelocities;

	double action_multiplier = 0.2;
	for(int i = 0; i < num_body_nodes*3; i++){
		mActions[i] = dart::math::clip(mActions[i]*action_multiplier, -0.7*M_PI, 0.7*M_PI);
	}

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		mModifiedTargetPositions.segment<3>(idx) += mActions.segment<3>(3*i);
	}

	// set pd gain action
	Eigen::VectorXd kp(mCharacter->GetSkeleton()->getNumDofs()), kv(mCharacter->GetSkeleton()->getNumDofs());

	kp.setConstant(500);
	kp.segment<6>(0).setZero();

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		if(mInterestedBodies[i] == "Spine1" || mInterestedBodies[i] == "FemurR" || mInterestedBodies[i] == "FemurL"){
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(1000);
		}
		else{
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(500);
		}
	}

	// kv = (kp.array() * 2).sqrt();
	kv = KV_RATIO*kp;
	mCharacter->SetPDParameters(kp,kv);
	for(int i=0;i<per;i+=2){
		Eigen::VectorXd torque = mCharacter->GetSPDForces(mModifiedTargetPositions,mModifiedTargetVelocities);
		for(int j=0;j<2;j++)
		{
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step();
		}
	}

	this->mControlCount++;
	this->mTimeElapsed += 1.0 / this->mControlHz;
}
void Controller::Reset(bool RSI)
{
	this->mFrame = 0;
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
		this->mTimeElapsed = dart::math::Random::uniform(0.0,this->mBVH->GetMaxTime() - 0.51 - 1.0/this->mControlHz);
		this->mControlCount = std::floor(this->mTimeElapsed*this->mControlHz);

	}
	else {
		this->mTimeElapsed = 0.0;
		this->mControlCount = 0;
	}
	mStartTime = mTimeElapsed;

	auto p_v_target = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mTimeElapsed);
	this->mTargetPositions = std::get<0>(target_tuple);
	this->mTargetVelocities = std::get<1>(target_tuple);
	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()*3);
	mActions.setZero();

	auto& skel = mCharacter->GetSkeleton();
	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);
	
	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;
	this->mTimeElapsed += 1.0 / this->mControlHz;
	this->mControlCount++;

}
bool
Controller::
IsTerminalState()
{
	if(mIsTerminal)
		return true;

	auto& skel = mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();

	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];
	Eigen::Vector3d root_v = skel->getBodyNode(0)->getCOMLinearVelocity();
	double root_v_norm = root_v.norm();
	Eigen::Vector3d root_pos_diff = this->mTargetPositions.segment<3>(3) - root_pos;

	skel->setPositions(this->mTargetPositions);
	skel->computeForwardKinematics(true, false, false);
	Eigen::Isometry3d root_diff = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	skel->setPositions(p);
	skel->computeForwardKinematics(true, false, false);

	Eigen::AngleAxisd root_diff_aa(root_diff.linear());
	double angle = RadianClamp(root_diff_aa.angle());

	// check nan
	if(dart::math::isNan(p)){
		// std::cout << "p nan" << std::endl;
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 3;
		return mIsTerminal;
	}
	if(dart::math::isNan(v)){
		// std::cout << "v nan" << std::endl;
		mIsNanAtTerminal = true;
		mIsTerminal = true;

		terminationReason = 4;
		return mIsTerminal;
	}
	//ET
	if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
			// std::cout << "root fall" << std::endl;
			mIsNanAtTerminal = false;
			// this->ComputeRootCOMDiff();
			mIsTerminal = true;
			terminationReason = 1;
	}
	else if(std::abs(root_pos[0]) > 4990){
			mIsNanAtTerminal = false;
			// this->ComputeRootCOMDiff();
			mIsTerminal = true;
			terminationReason = 9;
	}
	else if(std::abs(root_pos[2]) > 4990){
			mIsNanAtTerminal = false;
			// this->ComputeRootCOMDiff();
			mIsTerminal = true;
			terminationReason = 9;
	}
	else if(root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		 	mIsNanAtTerminal = false;
		 	// this->ComputeRootCOMDiff();
		 	mIsTerminal = true;
		 	terminationReason = 2;
	}
	else if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
			mIsNanAtTerminal = false;
			mIsTerminal = true;
			terminationReason = 5;
	}
	else if(this->mTimeElapsed > this->mBVH->GetMaxTime()-0.51-1.0/this->mControlHz){
		mIsNanAtTerminal = false;
		mIsTerminal = true;
		terminationReason =  8;
	}
	return mIsTerminal;
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
double
Controller::
GetReward()
{
	std::vector<double> ret = this->GetRewardByParts();
	return ret[0];
}

std::vector<double>
Controller::
GetRewardByParts()
{
	auto& skel = this->mCharacter->GetSkeleton();

	//Position Differences
	Eigen::VectorXd p_diff = skel->getPositionDifferences(this->mTargetPositions, skel->getPositions());

	//Velocity Differences
	Eigen::VectorXd v_diff = skel->getVelocityDifferences(this->mTargetVelocities, skel->getVelocities());

	Eigen::VectorXd p_diff_lower, v_diff_lower;
	Eigen::Vector3d root_ori_diff;
	Eigen::Vector3d root_av_diff;
	
	int index_torso   = skel->getBodyNode("Torso")->getParentJoint()->getIndexInSkeleton(0);	
	int num_lower_body_nodes = this->mRewardBodies.size();

	root_ori_diff = p_diff.segment<3>(index_torso);
	root_av_diff = v_diff.segment<3>(index_torso);


	p_diff_lower.resize(num_lower_body_nodes*3);
	v_diff_lower.resize(num_lower_body_nodes*3);

	for(int i = 0; i < num_lower_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mRewardBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		p_diff_lower.segment<3>(3*i) = p_diff.segment<3>(idx);
		v_diff_lower.segment<3>(3*i) = v_diff.segment<3>(idx);
	}

	//End-effector position and COM Differences
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	std::vector<Eigen::Isometry3d> ee_transforms;
	Eigen::VectorXd ee_diff(mEndEffectors.size()*3);
	Eigen::VectorXd ee_ori_diff(mEndEffectors.size()*3);
	Eigen::Vector3d com_diff, com_v_diff;


	for(int i=0;i<mEndEffectors.size();i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
	}
	
	com_diff = skel->getCOM();
	com_v_diff = skel->getCOMLinearVelocity();


	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	for(int i=0;i<mEndEffectors.size();i++){
		Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee_diff.segment<3>(3*i) = diff.translation();
		ee_ori_diff.segment<3>(3*i) = QuaternionToDARTPosition(Eigen::Quaterniond(diff.linear()));
	}

	com_diff -= skel->getCOM();
	com_v_diff -= skel->getCOMLinearVelocity();


	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);


	double foot_contact_match_l = 0;
	double foot_contact_match_r = 0;
	bool foot_l_contact = false;
	bool foot_r_contact = false;
	if( this->CheckCollisionWithGround("FootL")){
		foot_l_contact = true;
	}
	if( this->CheckCollisionWithGround("FootEndL")){
		foot_l_contact = true;
	}
	if(this->mRefFoot[0] > 0 && foot_l_contact){
		foot_contact_match_l = 1;
	}
	if(this->mRefFoot[0] < 0 && !foot_l_contact){
		foot_contact_match_l = 1;
	}


	if( this->CheckCollisionWithGround("FootR")){
		foot_r_contact = true;
	}
	if( this->CheckCollisionWithGround("FootEndR")){
		foot_r_contact = true;
	}
	if(this->mRefFoot[1] > 0 && foot_r_contact){
		foot_contact_match_r = 1;
	}
	if(this->mRefFoot[1] < 0 && !foot_r_contact){
		foot_contact_match_r = 1;
	}

	double scale = 1.0;
	double sig_p = 0.1 * scale; 		// 2
	double sig_v = 1.0 * scale;		// 3
	double sig_com = 0.3 * scale;		// 4
	double sig_ee = 0.3 * scale;		// 8

	double sig_com_v = 0.2 * scale;		// 5
	double sig_ori = 0.8 * scale;		// 6
	double sig_av = 4.0 * scale;		// 7
	double sig_ee_ori = 1.2 * scale;	// 9

	w_p = 0.35;
	w_v = 0.1;


	w_com = 0.3;
	double w_com_v = 0.00;
	w_root_ori = 0.00;
	w_root_av = 0.00;

	w_ee = 0.25;
	double w_ee_ori = 0.0;

	// w_goal = 0.0;
	// double w_p_g = 0.0;

	double r_p = exp_of_squared(p_diff_lower,sig_p);
	double r_v = exp_of_squared(v_diff_lower,sig_v);

	double r_ori = exp_of_squared(root_ori_diff, sig_ori);
	double r_av = exp_of_squared(root_av_diff, sig_av);
	// double r_p_g = exp_of_squared(p_global_diff, sig_p_g);
	double r_ee = exp_of_squared(ee_diff,sig_ee);
	double r_ee_ori = exp_of_squared(ee_ori_diff,sig_ee_ori);
	double r_com = exp_of_squared(com_diff,sig_com);
	double r_com_v = exp_of_squared(com_v_diff,sig_com_v);
	// double r_mass = exp_of_squared(mass_diff,sig_mass);
	// double r_goal = exp_of_squared(local_goal,sig_goal);
	// double r_tot = w_p*r_p 
	// 				+ w_v*r_v 
	// 				+ w_com*r_com
	// 				+ w_ee*r_ee;
	double r_tot = r_p*r_v*r_com*r_ee;
	// double r_tot = (w_p*r_p 
	// 				+ w_v*r_v 
	// 				+ w_root_ori*r_ori 
	// 				+ w_root_av*r_av
	// 				+ w_ee*r_ee 
	// 				+ w_goal*r_goal)	
	// 				* r_com;

	std::vector<double> ret;
	ret.clear();
	if(dart::math::isNan(r_tot)){
		ret.resize(8, 0.0);
		return ret;
	}

	ret.push_back(r_tot);

	ret.push_back(r_p);
	ret.push_back(r_v);
	ret.push_back(r_com);
	ret.push_back(r_ee);

	return ret;
}
