#include "Controller.h"
#include "Character.h"
#include <boost/filesystem.hpp>
#include <fstream>

namespace DPhy
{	
Controller::Controller(std::string motion)
	:mTimeElapsed(0.0),mControlHz(30),mSimulationHz(600),mControlCount(0),
	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{
	this->mSimPerCon = mSimulationHz / mControlHz;

	this->mWorld = std::make_shared<dart::simulation::World>();
	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81,0));

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml"));
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);

	std::string path = std::string(CAR_DIR)+std::string("/character/humanoid_new.xml");
	this->mCharacter = new DPhy::Character(path);
	this->mCharacter->LoadBVHMap(path);
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
	this->mCGG = collisionEngine->createCollisionGroup(this->mGround.get());

	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()*3);
	mActions.setZero();

	mEndEffectors.clear();
	mEndEffectors.push_back("FootR");
	mEndEffectors.push_back("FootL");
	mEndEffectors.push_back("HandL");
	mEndEffectors.push_back("HandR");
	mEndEffectors.push_back("Head");

	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	this->SetReference(motion);
	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();
	
	this->torques.clear();
}
void 
Controller::
SetReference(std::string motion) 
{
	this->mBVH = new BVH();
	std::string path = std::string(CAR_DIR) + std::string("/motion/") + motion + std::string(".bvh");
	this->mBVH->Parse(path);
	this->mCharacter->InitializeBVH(this->mBVH);
	std::cout << "mapping done" << std::endl;
}
void 
Controller::
Step()
{
	if(IsTerminalState())
		return;
	
	// set action target pos
	int num_body_nodes = this->mInterestedBodies.size();

	auto p_v_target = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mTimeElapsed);
	this->mTargetPositions = std::get<0>(p_v_target);
	this->mTargetVelocities = std::get<1>(p_v_target);
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
	for(int i = 0; i < this->mSimPerCon; i += 2){
		Eigen::VectorXd torque = mCharacter->GetSPDForces(mModifiedTargetPositions,mModifiedTargetVelocities);
		for(int j = 0; j < 2; j++)
		{
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step();
		}
	}
	this->mControlCount++;
	this->mTimeElapsed += 1.0 / this->mControlHz;

	this->UpdateReward();
	this->UpdateTerminalInfo();
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
	Eigen::Vector3d root_ori_diff;
	Eigen::Vector3d root_av_diff;
	
	int num_reward_body_nodes = this->mRewardBodies.size();

	root_ori_diff = p_diff.segment<3>(0);
	root_av_diff = v_diff.segment<3>(0);

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


	for(int i=0;i<mEndEffectors.size();i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
	}
	
	com_diff = skel->getCOM();

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	for(int i=0;i<mEndEffectors.size();i++){
		Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee_diff.segment<3>(3*i) = diff.translation();
	}

	com_diff -= skel->getCOM();

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

	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_v = exp_of_squared(v_diff_reward,sig_v);
	double r_ee = exp_of_squared(ee_diff,sig_ee);
	double r_com = exp_of_squared(com_diff,sig_com);

	double r_tot = r_p*r_v*r_com*r_ee;


	mRewardParts.clear();
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(5, 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(r_p);
		mRewardParts.push_back(r_v);
		mRewardParts.push_back(r_com);
		mRewardParts.push_back(r_ee);
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
		return mIsTerminal;
	}
	if(dart::math::isNan(v)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 4;
		return mIsTerminal;
	}
	//characterConfigration
	if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
		mIsTerminal = true;
		terminationReason = 1;
	}
	else if(root_diff.translation().norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		 mIsTerminal = true;
		 terminationReason = 2;
	}
	else if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 5;
	}
	else if(this->mTimeElapsed > this->mBVH->GetMaxTime()-1.0/this->mControlHz){
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

	auto p_v_target = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mTimeElapsed);
	mTargetPositions = p_v_target.first;
	mTargetVelocities = p_v_target.second;

	for(int i=0;i<this->mSimPerCon;i++)
	{
		skel->setPositions(mTargetPositions);
		skel->setVelocities(mTargetVelocities);
		skel->computeForwardKinematics(true, true, false);
	}
	this->mControlCount++;
	this->mTimeElapsed += 1.0 / this->mControlHz;
	return true;
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
		this->mTimeElapsed = dart::math::Random::uniform(0.0,this->mBVH->GetMaxTime() - 10 /this->mControlHz);
		this->mControlCount = std::floor(this->mTimeElapsed*this->mControlHz);
	}
	else {
		this->mTimeElapsed = 0.0;
		this->mControlCount = 0;
	}

	auto p_v_target = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mTimeElapsed);
	this->mTargetPositions = std::get<0>(p_v_target);
	this->mTargetVelocities = std::get<1>(p_v_target);

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;
	this->mTimeElapsed += 1.0 / this->mControlHz;
	this->mControlCount++;

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

	ret.resize((num_ee)*9+12);
	for(int i=0;i<num_ee;i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
		ret.segment<6>(6*i) << rot, transform.translation();
	}


	for(int i=0;i<num_ee;i++)
	{
	    int idx = skel->getBodyNode(mEndEffectors[i])->getParentJoint()->getIndexInSkeleton(0);
		ret.segment<3>(6*num_ee + 3*i) << vel.segment<3>(idx);
	}

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
	Eigen::Vector3d root_angular_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getAngularVelocity();
	Eigen::Vector3d root_linear_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getCOMLinearVelocity();
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
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	int num_body_nodes = mInterestedBodies.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	std::vector<Eigen::VectorXd> tp_vec;
	tp_vec.clear();
	std::vector<int> tp_times;
	tp_times.clear();
	tp_times.push_back(0);
	
	for(auto dt : tp_times){
		int t = std::max(0, this->mControlCount + dt);
		auto target_tuple = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(this->mBVH, t * this->mSimPerCon);

		Eigen::VectorXd p = std::get<0>(target_tuple);
		Eigen::VectorXd v = std::get<1>(target_tuple);
		tp_vec.push_back(this->GetEndEffectorStatePosAndVel(p, v));
	}

	Eigen::VectorXd tp_concatenated;
	tp_concatenated.resize(tp_vec.size()*tp_vec[0].rows());
	for(int i = 0; i < tp_vec.size(); i++){
		tp_concatenated.segment(i*tp_vec[0].rows(), tp_vec[0].rows()) = tp_vec[i];
	}

	Eigen::VectorXd p,v;
	p.resize(p_save.rows()-6);
	p = p_save.tail(p_save.rows()-6);
	v = v_save/10.0;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	double root_height = skel->getRootBodyNode()->getCOM()[1];

	const dart::dynamics::BodyNode *bnL, *bnEL, *bnR, *bnER;
	bnL = skel->getBodyNode("FootL");
	bnEL = skel->getBodyNode("FootEndL");
	bnR = skel->getBodyNode("FootR");
	bnER = skel->getBodyNode("FootEndR");

	Eigen::Vector3d p0 = Eigen::Vector3d(0.04, -0.025, -0.065);
	Eigen::Vector3d p1 = Eigen::Vector3d(-0.04, -0.025, -0.065);
	Eigen::Vector3d p2 = Eigen::Vector3d(0.04, -0.025, 0.035);
	Eigen::Vector3d p3 = Eigen::Vector3d(-0.04, -0.025, 0.035);

	Eigen::Vector3d p0_l = bnL->getWorldTransform()*p0;
	Eigen::Vector3d p1_l = bnL->getWorldTransform()*p1;
	Eigen::Vector3d p2_l = bnEL->getWorldTransform()*p2;
	Eigen::Vector3d p3_l = bnEL->getWorldTransform()*p3;

	Eigen::Vector3d p0_r = bnR->getWorldTransform()*p0;
	Eigen::Vector3d p1_r = bnR->getWorldTransform()*p1;
	Eigen::Vector3d p2_r = bnER->getWorldTransform()*p2;
	Eigen::Vector3d p3_r = bnER->getWorldTransform()*p3;

	Eigen::VectorXd foot_corner_heights;
	foot_corner_heights.resize(8);
	foot_corner_heights << p0_l[1], p1_l[1], p2_l[1], p3_l[1], 
							p0_r[1], p1_r[1], p2_r[1], p3_r[1];
	foot_corner_heights *= 10;

	Eigen::VectorXd state;
	state.resize(p.rows()+v.rows()+tp_concatenated.rows()+1+1+8);
	state<<p, v, tp_concatenated, up_vec_angle,
			root_height, foot_corner_heights;

	return state;
}
}