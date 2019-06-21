#include "HumanoidController.h"
#include "Ground.h"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <omp.h>
#include <tuple>
#include "CharacterConfigurations.h"

#define ACTION_TYPE 3 // 0 : delta, 1 : absolute quaternion
#define USE_RNN 1

namespace DPhy
{
HumanoidController::
HumanoidController(bool use_trajectory, bool use_terminal, bool discrete_reference)
	:Controller(),mTimeElapsed(0.0),mControlHz(30),mSimulationHz(600),mActions(Eigen::VectorXd::Zero(0)),mUpdated(false)
	,w_p(0.2),w_v(0.1),w_ee(0.2),w_com(0.25),w_root_ori(0.1),w_root_av(0.05),w_goal(0.1)
	,mUseTrajectory(use_trajectory),mUseDiscreteReference(discrete_reference),mUseTerminal(use_terminal)
	,terminationReason(-1),mIsNanAtTerminal(false),mGenerator(std::chrono::high_resolution_clock::now().time_since_epoch().count())
	,mCurrentReferenceManagerIndex(0)
{
	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	this->AddCharacter(new Ground());
	// std::cout << this->mCharacters["Ground"]->GetSkeleton()->getBodyNode(0)->getFrictionCoeff() << std::endl;
	// std::cout << this->mCharacters["Ground"]->GetSkeleton()->getBodyNode(0)->getRestitutionCoeff() << std::endl;
	this->mCharacters["Ground"]->GetSkeleton()->getBodyNode(0)->setFrictionCoeff(1.0);
	// this->mCharacters["Ground"]->GetSkeleton()->getBodyNode(0)->setRestitutionCoeff(0.0);
	
	// mHumanoid = new Humanoid(std::string(DPHY_DIR)+std::string("/character/humanoid3.xml"));
	// mHumanoid = new Humanoid(std::string(DPHY_DIR)+std::string("/character/humanoid4.xml"));
#ifdef NEW_JOINTS
	mHumanoid = new Humanoid(std::string(DPHY_DIR)+std::string("/character/humanoid_new.xml"));
#else
	mHumanoid = new Humanoid(std::string(DPHY_DIR)+std::string("/character/humanoid7_nolimit.xml"));
#endif
	this->AddCharacter(mHumanoid);

	this->mPositionUpperLimits = this->mHumanoid->GetSkeleton()->getPositionUpperLimits();
	this->mPositionLowerLimits = this->mHumanoid->GetSkeleton()->getPositionLowerLimits();
	this->mActionRange = (this->mPositionUpperLimits - this->mPositionLowerLimits)*0.1;
	// std::cout << this->mPositionUpperLimits.transpose() << std::endl;
	// std::cout << this->mPositionLowerLimits.transpose() << std::endl;
	// std::cout << this->mActionRange.transpose() << std::endl;

	// int index_chest = mHumanoid->GetSkeleton()->getBodyNode("Chest")->getParentJoint()->getIndexInSkeleton(0);
	int index_femur_r = mHumanoid->GetSkeleton()->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	int index_femur_l = mHumanoid->GetSkeleton()->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int index_tibia_r = mHumanoid->GetSkeleton()->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);
	int index_tibia_l = mHumanoid->GetSkeleton()->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	// int index_talus_r = mHumanoid->GetSkeleton()->getBodyNode("TalusR")->getParentJoint()->getIndexInSkeleton(0);
	// int index_talus_l = mHumanoid->GetSkeleton()->getBodyNode("TalusL")->getParentJoint()->getIndexInSkeleton(0);

	// this->mHumanoid->GetSkeleton()->getBodyNode("FootL")->setRestitutionCoeff(0.0);
	// this->mHumanoid->GetSkeleton()->getBodyNode("FootR")->setRestitutionCoeff(0.0);

	this->mMaxJoint = Eigen::VectorXd::Constant(mHumanoid->GetSkeleton()->getNumDofs(), -20.);
	this->mMinJoint = Eigen::VectorXd::Constant(mHumanoid->GetSkeleton()->getNumDofs(), 20.);


	Eigen::VectorXd kp(mHumanoid->GetSkeleton()->getNumDofs()), kv(mHumanoid->GetSkeleton()->getNumDofs());

	kp.setZero();
	// kp.segment<3>(index_chest) = Eigen::Vector3d::Constant(1000);

	// kp.segment<3>(index_femur_r) = Eigen::Vector3d::Constant(300);
	// kp.segment<3>(index_femur_l) = Eigen::Vector3d::Constant(300);

	// kp.segment<3>(index_tibia_r) = Eigen::Vector3d::Constant(300);
	// kp.segment<3>(index_tibia_l) = Eigen::Vector3d::Constant(300);

	// kp.segment<3>(index_talus_r) = Eigen::Vector3d::Constant(300);
	// kp.segment<3>(index_talus_l) = Eigen::Vector3d::Constant(300);


	kv = kp * 0.1;
	mHumanoid->SetPDParameters(kp,kv);

	if(useBall) mThrowingBall= new ThrowingBall(this->mWorld, this->mHumanoid->GetSkeleton());

//    this->AddCharacter(new ThrowingBall(mHumanoid, 0.2, 0.5 ));

#if (USE_RNN == 1)
	this->mReferenceManagers.resize(REFERENCE_MANAGER_COUNT);
	for(int i = 0; i < REFERENCE_MANAGER_COUNT; i++)
		this->mReferenceManagers[i] = new ReferenceManager(mHumanoid, mControlHz);
#elif (USE_RNN == 0)
	// this->mMotionGenerator = new MotionGenerator(mHumanoid);
	// this->mMotionGenerator->addBVHs(std::string(DPHY_DIR)+"/motion/included_motions");
	// this->mMotionGenerator->Initialize();
	this->mBVH = new BVH();
	// std::string motionfilename = std::string(DPHY_DIR)+std::string("/motion/dribble_test3.bvh");
	std::string motionfilename = std::string(DPHY_DIR)+std::string("/motion/dribble_test.bvh");
	this->mBVH->Parse(motionfilename);
	this->mHumanoid->InitializeBVH(this->mBVH);
#endif

#ifdef NEW_JOINTS
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
#else
	mInterestedBodies.clear();
	// mInterestedBodies.push_back("Spine");
	mInterestedBodies.push_back("Spine1");
	// mInterestedBodies.push_back("Spine2");
	// mInterestedBodies.push_back("Neck");
	mInterestedBodies.push_back("Head");


	// mInterestedBodies.push_back("ShoulderL");
	mInterestedBodies.push_back("ForeArmL");
	mInterestedBodies.push_back("ArmL");
	mInterestedBodies.push_back("HandL");

	// mInterestedBodies.push_back("ShoulderR");
	mInterestedBodies.push_back("ForeArmR");
	mInterestedBodies.push_back("ArmR");
	mInterestedBodies.push_back("HandR");

	mInterestedBodies.push_back("FemurL");
	mInterestedBodies.push_back("TibiaL");
	// mInterestedBodies.push_back("TalusL");
	mInterestedBodies.push_back("FootL");
	mInterestedBodies.push_back("FootEndL");

	mInterestedBodies.push_back("FemurR");
	mInterestedBodies.push_back("TibiaR");
	// mInterestedBodies.push_back("TalusR");
	mInterestedBodies.push_back("FootR");
	mInterestedBodies.push_back("FootEndR");



	mRewardBodies.clear();
	mRewardBodies.push_back("Torso");
	mRewardBodies.push_back("FemurR");
	mRewardBodies.push_back("TibiaR");
	// mRewardBodies.push_back("TalusR");
	mRewardBodies.push_back("FootR");

	mRewardBodies.push_back("FemurL");
	mRewardBodies.push_back("TibiaL");
	// mRewardBodies.push_back("TalusL");
	mRewardBodies.push_back("FootL");

	mRewardBodies.push_back("Spine1");
	mRewardBodies.push_back("Head");

	mRewardBodies.push_back("ForeArmL");
	mRewardBodies.push_back("ArmL");
	mRewardBodies.push_back("HandL");

	mRewardBodies.push_back("ForeArmR");
	mRewardBodies.push_back("ArmR");
	mRewardBodies.push_back("HandR");

#endif


	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	// std::cout << dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->getBoxedLcpSolver()->getType() << std::endl;
	this->mCGL = collisionEngine->createCollisionGroup(this->mWorld->getSkeleton("Humanoid")->getBodyNode("FootL"));
	this->mCGR = collisionEngine->createCollisionGroup(this->mWorld->getSkeleton("Humanoid")->getBodyNode("FootR"));
	this->mCGEL = collisionEngine->createCollisionGroup(this->mWorld->getSkeleton("Humanoid")->getBodyNode("FootEndL"));
	this->mCGER = collisionEngine->createCollisionGroup(this->mWorld->getSkeleton("Humanoid")->getBodyNode("FootEndR"));
	this->mCGG = collisionEngine->createCollisionGroup(this->mWorld->getSkeleton("Ground").get());

#if (ACTION_TYPE == 0)
	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()*(3+3+3));
#elif(ACTION_TYPE == 1)
	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()*(4+3));
#elif(ACTION_TYPE == 2)
	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()*(4+3));
#elif(ACTION_TYPE == 3)
	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()*(3));
#endif
	mActions.setZero();

	mEndEffectors.clear();
	mEndEffectors.push_back("FootR");
	mEndEffectors.push_back("FootL");
	mEndEffectors.push_back("HandL");
	mEndEffectors.push_back("HandR");
	mEndEffectors.push_back("Head");

	int dof = this->mHumanoid->GetSkeleton()->getNumDofs(); 
	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	// Reset(false);
	this->mRootCOMAtTerminal.setZero();
	this->mRootCOMAtTerminalRef.setZero();


	this->mGoal.setZero();

	this->mIsTerminal = false;
	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();
	torques.clear();
}

void HumanoidController::FollowBvh(){	
	if(IsTerminalState())
		return;
	auto& skel = mHumanoid->GetSkeleton();
	this->mControlCount++;
	int per = mSimulationHz/mControlHz;

	// auto p_v_target2 = mHumanoid->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mTimeElapsed);
	// mTargetPositions = p_v_target2.first;
	// mTargetVelocities = p_v_target2.second;

	// Eigen::VectorXd cur_pos = skel->getPositions();
	// Eigen::VectorXd cur_vel = skel->getVelocities();

	// mHumanoid->GetSkeleton()->setPositions(mTargetPositions);
	// mHumanoid->GetSkeleton()->setVelocities(mTargetVelocities);
	// std::cout << std::endl;
	// mHumanoid->GetSkeleton()->computeForwardKinematics(true,false,false);
	// std::cout << mHumanoid->GetSkeleton()->getCOMLinearVelocity().transpose() << std::endl;
	// dynamic_cast<dart::dynamics::FreeJoint*>(mHumanoid->GetSkeleton()->getRootJoint())->setLinearVelocity(mTargetVelocities.segment<3>(3));
	// mHumanoid->GetSkeleton()->computeForwardKinematics(true,false,false);
	// std::cout << mHumanoid->GetSkeleton()->getCOMLinearVelocity().transpose() << std::endl;

	// std::cout << mTargetVelocities.segment<3>(3).transpose() << std::endl;
	// mTargetVelocities = mHumanoid->GetSkeleton()->getVelocities();
	// std::cout << mTargetVelocities.segment<3>(3).transpose() << std::endl;

	// mHumanoid->GetSkeleton()->setPositions(cur_pos);
	// mHumanoid->GetSkeleton()->setVelocities(cur_vel);
	// mHumanoid->GetSkeleton()->computeForwardKinematics(true,false,false);


	for(int i=0;i<per;i++)
	{
		// Debug
		// Eigen::Vector3d dd = 90./180.*M_PI*Eigen::Vector3d::UnitY();
		// mTargetPositions.segment<3>(0) = dd;
		this->Record();

		mTimeElapsed += 1.0/mSimulationHz;
		// auto p_v_target = mHumanoid->GetTargetPositionsAndVelocitiesFromBVH(mBVH, mTimeElapsed);
		// auto p_v_target = this->mMotionGenerator->getMotion(mTimeElapsed);
		// this->mTargetPositions = p_v_target.first;
		// this->mTargetVelocities = p_v_target.second;
		skel->setPositions(mTargetPositions);
		skel->setVelocities(mTargetVelocities);
		skel->computeForwardKinematics(true, true, false);
		// DPhy::Controller::Step(true);
	}
	// mTimeElapsed += 1. / mControlHz;
}
 
void HumanoidController::UpdateReferenceDataForCurrentTime(){
	if(this->mUpdated)
		return;
#if (USE_RNN == 1)
	if(this->mUseTrajectory){
		if(this->mUseDiscreteReference){
			auto target_tuple = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getMotion(this->mControlCount);	
			this->mTargetPositions = std::get<0>(target_tuple);
			this->mTargetVelocities = std::get<1>(target_tuple);
			this->mGoal = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getGoal(this->mControlCount);
			this->mRefFoot = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getFootContacts(this->mControlCount);
		}
		else{
			auto target_tuple = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getMotion(this->mTimeElapsed);	
			this->mTargetPositions = std::get<0>(target_tuple);
			this->mTargetVelocities = std::get<1>(target_tuple);
			this->mGoal = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getGoal(this->mTimeElapsed);
			this->mRefFoot = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getFootContacts(this->mTimeElapsed);
		}
	}
	else{
		if(this->mUseDiscreteReference){
			auto target_tuple = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getMotion(0);	
			this->mTargetPositions = std::get<0>(target_tuple);
			this->mTargetVelocities = std::get<1>(target_tuple);
			this->mGoal = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getGoal();
			this->mRefFoot = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getFootContacts(0);
		}
		else{
			auto target_tuple = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getMotion(0.0);	
			this->mTargetPositions = std::get<0>(target_tuple);
			this->mTargetVelocities = std::get<1>(target_tuple);
			this->mGoal = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getGoal();
			this->mRefFoot = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getFootContacts(0.0);
		}
	}
#elif (USE_RNN == 0)
	auto target_tuple = this->mHumanoid->GetTargetPositionsAndVelocitiesFromBVH(this->mBVH, this->mTimeElapsed);
	this->mTargetPositions = std::get<0>(target_tuple);
	this->mTargetVelocities = std::get<1>(target_tuple);
	// this->mGoal = this->mReferenceManager->getGoal(this->mTimeElapsed);
	// this->mRefFoot = this->mReferenceManager->getFootContacts(this->mTimeElapsed);		
#endif
	this->mUpdated = true;
}

void HumanoidController::FollowReference(){	
	if(IsTerminalState())
		return;

	auto& skel = this->mHumanoid->GetSkeleton();
	this->mControlCount++;
	int per = this->mSimulationHz/this->mControlHz;

	for(int i=0;i<per;i++)
	{
		this->Record();

		this->mTimeElapsed += 1.0/this->mSimulationHz;
		this->UpdateReferenceDataForCurrentTime();
		this->mModifiedTargetPositions = this->mTargetPositions;
		if(this->mUseDiscreteReference)
			this->mTargetPositions = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getOriginPositions(this->mControlCount);
		else
			this->mTargetPositions = this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getOriginPositions(this->mTimeElapsed);
		// if(this->mRefFoot[0] < 0.5){
		// 	skel->setPositions(this->mModifiedTargetPositions);
		// 	skel->computeForwardKinematics(true, false, false);
		// 	Eigen::VectorXd newPose = solveIK(skel, "TalusL", Eigen::Vector3d(0.0, 0.1, 0.0));
		// 	this->mModifiedTargetPositions = newPose;
		// }
		// if(this->mRefFoot[1] < 0.5){
		// 	skel->setPositions(this->mModifiedTargetPositions);
		// 	skel->computeForwardKinematics(true, false, false);
		// 	Eigen::VectorXd newPose = solveIK(skel, "TalusR", Eigen::Vector3d(0.0, 0.1, 0.0));
		// 	this->mModifiedTargetPositions = newPose;
		// }
		skel->setPositions(this->mModifiedTargetPositions);
		skel->setVelocities(this->mTargetVelocities);
		skel->computeForwardKinematics(true, true, false);

        if(useBall)
        {
            mThrowingBall->step();
            mThrowingBall->createNewBallPeriodically();
            mThrowingBall->deleteBallAutomatically();
//            mThrowingBall->step();

//            std::cout<<"followReference/ ball # "<<mThrowingBall->mBalls.size()<<std::endl;
        }
		// DPhy::Controller::Step(true);
	}

	// // goal
	// Eigen::Vector3d local_goal = skel->getRootBodyNode()->getTransform().inverse() * this->mGoal;
	// std::cout << local_goal[0] << ", " << local_goal[2] << std::endl;
	// mTimeElapsed += 1. / mControlHz;
}

void
HumanoidController::
SetReferenceToTarget(const Eigen::VectorXd& ref_cur, const Eigen::VectorXd& ref_next){
	this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->setTargetMotion(ref_cur, ref_next);
}

void
HumanoidController::
SetReferenceTrajectory(const Eigen::MatrixXd& trajectory){
    this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->setTrajectory(trajectory);
	// this->mReferenceManager->saveReferenceTrajectory("trajectory.txt");
}

void
HumanoidController::
SaveReferenceTrajectory(){
    this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->saveReferenceTrajectory("trajectory.txt");
}

Eigen::VectorXd SkelSlerp(const Eigen::VectorXd& s1, const Eigen::VectorXd& s2, double t){
	Eigen::VectorXd ret = Eigen::VectorXd::Zero(s1.rows());

	ret.segment<3>(0) = QuaternionToDARTPosition(DARTPositionToQuaternion(s1.segment<3>(0)).slerp(t, DARTPositionToQuaternion(s2.segment<3>(0))));
	ret.segment<3>(3) = s1.segment<3>(3) * (1-t) + s2.segment<3>(3) * t;

	for(int i = 6; i < s1.rows(); i+=3){
		ret.segment<3>(i) = QuaternionToDARTPosition(DARTPositionToQuaternion(s1.segment<3>(i)).slerp(t, DARTPositionToQuaternion(s2.segment<3>(i))));
	}
	return ret;
}


void
HumanoidController::
AddReference(const Eigen::VectorXd& ref, int index){
	if(index == -1)
		index = this->mCurrentReferenceManagerIndex;
	this->mReferenceManagers[index]->addPosition(ref);
	// Eigen::VectorXd converted = this->mReferenceManager->convertRNNMotion(ref);
	// this->mTargetPositions = converted;
	// Eigen::VectorXd cconverted = rootDecomposition(this->mHumanoid->GetSkeleton(), converted);
	// std::cout << ref.transpose() << std::endl;
	// std::cout << cconverted.transpose() << std::endl;
}

void
HumanoidController::
UpdateMax(){
	Eigen::VectorXd pos = this->mTargetPositions;
	for(int i = 0; i < pos.rows(); i++){
		if(pos[i] > this->mMaxJoint[i]){
			this->mMaxJoint[i] = pos[i];
		}
	}
}

void
HumanoidController::
UpdateMin(){
	Eigen::VectorXd pos = this->mTargetPositions;
	for(int i = 0; i < pos.rows(); i++){
		if(pos[i] < this->mMinJoint[i]){
			this->mMinJoint[i] = pos[i];
		}
	}
}

ThrowingBall*
HumanoidController::
getThrowingBall()
{
    return mThrowingBall;
}

void
HumanoidController::
createNewBall()
{
    mThrowingBall->createNewBall();
}

void
HumanoidController::
Step(bool record)
{
	this->UpdateReferenceDataForCurrentTime();
	int per = mSimulationHz/mControlHz;
	if(IsTerminalState())
		return;
	
	// set action target pos
	int num_body_nodes = this->mInterestedBodies.size();
	double pd_gain_offset = 0;
#if (ACTION_TYPE == 0)
	mModifiedTargetPositions = this->mTargetPositions;
	mModifiedTargetVelocities = this->mTargetVelocities;
	// mModifiedTargetPositions = mHumanoid->GetSkeleton()->getPositions();
	double action_multiplier = 0.2;
	for(int i = 0; i < num_body_nodes*3; i++){
		mActions[i] = dart::math::clip(mActions[i]*action_multiplier, -0.7*M_PI, 0.7*M_PI);
	}

	for(int i = num_body_nodes*3; i < num_body_nodes*3*2; i++){
		mActions[i] = mActions[i]*action_multiplier;
	}

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mHumanoid->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		mModifiedTargetPositions.segment<3>(idx) += mActions.segment<3>(3*i);
		mModifiedTargetVelocities.segment<3>(idx) += mActions.segment<3>(num_body_nodes*3+3*i);
	}

	pd_gain_offset = 3*2*num_body_nodes;
#elif (ACTION_TYPE == 1)
	mModifiedTargetPositions = this->mTargetPositions;
	for(int i = 0; i < num_body_nodes*4; i++){
		mActions[i] = dart::math::clip(mActions[i], -1., 1.);
	}
	for(int i = 0; i < num_body_nodes; i++){
		int idx = mHumanoid->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		Eigen::Quaterniond q;
		// normalize q input
		// actually, implicitly done by coversion from quat to aa in QuaternionToDARTPosition
		if(mActions.segment<4>(4*i).norm() < 1e-5){
			q.setIdentity();
		}
		else{
			q.coeffs() = mActions.segment<4>(4*i).normalized();
		}
		QuaternionNormalize(q);
		mModifiedTargetPositions.segment<3>(idx) = QuaternionToDARTPosition(q);
	}
	pd_gain_offset = 4*num_body_nodes;

#elif (ACTION_TYPE == 2)
	mModifiedTargetPositions = mHumanoid->GetSkeleton()->getPositions();
	for(int i = 0; i < num_body_nodes*4; i++){
		mActions[i] = dart::math::clip(mActions[i], -1., 1.);
	}
	for(int i = 0; i < num_body_nodes; i++){
		int idx = mHumanoid->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		Eigen::Quaterniond q;
		// normalize q input
		// actually, implicitly done by coversion from quat to aa in QuaternionToDARTPosition
		if(mActions.segment<4>(4*i).norm() < 1e-5){
			q.setIdentity();
		}
		else{
			q.coeffs() = mActions.segment<4>(4*i).normalized();
		}
		QuaternionNormalize(q);
		mModifiedTargetPositions.segment<3>(idx) = QuaternionToDARTPosition(DARTPositionToQuaternion(mModifiedTargetPositions.segment<3>(idx))*q);
	}
	pd_gain_offset = 4*num_body_nodes;
#elif (ACTION_TYPE == 3)
	mModifiedTargetPositions = this->mTargetPositions;
	mModifiedTargetVelocities = this->mTargetVelocities;
	// mModifiedTargetPositions = mHumanoid->GetSkeleton()->getPositions();
	double action_multiplier = 0.2;
	for(int i = 0; i < num_body_nodes*3; i++){
		mActions[i] = dart::math::clip(mActions[i]*action_multiplier, -0.7*M_PI, 0.7*M_PI);
	}

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mHumanoid->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		mModifiedTargetPositions.segment<3>(idx) += mActions.segment<3>(3*i);
	}
#endif

	// set pd gain action
	Eigen::VectorXd kp(mHumanoid->GetSkeleton()->getNumDofs()), kv(mHumanoid->GetSkeleton()->getNumDofs());

	kp.setConstant(500);
	kp.segment<6>(0).setZero();

#if (ACTION_TYPE==3)
	for(int i = 0; i < num_body_nodes; i++){
		int idx = mHumanoid->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		if(mInterestedBodies[i] == "Spine1" || mInterestedBodies[i] == "FemurR" || mInterestedBodies[i] == "FemurL"){
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(1000);
		}
		else{
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(500);
		}
	}
#else
	for(int i = pd_gain_offset; i < num_body_nodes*3+pd_gain_offset; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.1+1.0, 0.2, 1.8);
	}

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mHumanoid->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		if(mInterestedBodies[i] == "Spine1" || mInterestedBodies[i] == "FemurR" || mInterestedBodies[i] == "FemurL"){
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(1000).array() * mActions.segment<3>(3*i+pd_gain_offset).array();
		}
		else{
			kp.segment<3>(idx) = Eigen::Vector3d::Constant(500).array() * mActions.segment<3>(3*i+pd_gain_offset).array();
		}
	}
#endif
	// kv = (kp.array() * 2).sqrt();
	kv = KV_RATIO*kp;
	mHumanoid->SetPDParameters(kp,kv);

	Eigen::VectorXd p_save = mHumanoid->GetSkeleton()->getPositions();
	int dof = mHumanoid->GetSkeleton()->getNumDofs();
	for(int i=0;i<per;i+=2){
		// Eigen::VectorXd cur_target = SkelSlerp(p_save, mModifiedTargetPositions, (1.+i)/per);
		// Eigen::VectorXd cur_target = mModifiedTargetPositions;
		Eigen::VectorXd torque = mHumanoid->GetSPDForces(mModifiedTargetPositions,mModifiedTargetVelocities);
		// if(i==0) std::cout << torque.transpose() << std::endl;
		// Eigen::VectorXd torque = mHumanoid->GetSPDForces(mModifiedTargetPositions,Eigen::VectorXd::Zero(this->mTargetVelocities.rows()));

		// for(int k=0;k<torque.rows();k++){
		// 	if(std::abs(torque[k])>1000)
		// 		std::cout << "torque exceed limmits at " << k << ", " << torque[k] << std::endl;
		// }

		for(int j=0;j<2;j++)
		{
			mHumanoid->GetSkeleton()->setForces(torque);

			if(useBall)
            {
//			    mThrowingBall->createNewBallPeriodically();
			    mThrowingBall->deleteBallAutomatically();
            }

			DPhy::Controller::Step(record);
		}
	}

	this->mTimeElapsed += 1.0 / this->mControlHz;
	this->mControlCount++;
	this->mUpdated = false;
}

void HumanoidController::Reset(bool RSI)
{
	DPhy::Controller::Reset(RSI);

	//RSI
#if (USE_RNN == 1)
	if(this->mUseTrajectory){
		if(RSI){
			this->mTimeElapsed = dart::math::Random::uniform(0.0,this->GetMaxTime());
			this->mControlCount = std::floor(this->mTimeElapsed*this->mControlHz);
		}
		else{
			this->mTimeElapsed = 0.0;
			this->mControlCount = 0;
		}
		this->mStartTime = this->mTimeElapsed;

	}
	else{
		this->mTimeElapsed = 0.0;
		this->mControlCount = 0;
		this->mStartTime = this->mTimeElapsed;
	}
#elif (USE_RNN == 0)
		if(RSI){
			this->mTimeElapsed = dart::math::Random::uniform(0.0,this->mBVH->GetMaxTime() - 0.51 - 1.0/this->mControlHz);
			this->mControlCount = std::floor(this->mTimeElapsed*this->mControlHz);
		}
		else{
			this->mTimeElapsed = 0.0;
			this->mControlCount = 0;
		}

#endif
	this->mUpdated = false;
	this->UpdateReferenceDataForCurrentTime();

	// change root position and orientation
	// double root_x_diff = dart::math::Random::uniform(-0.1, 0.1);
	// double root_z_diff = dart::math::Random::uniform(-0.1, 0.1);
	// double root_angle_diff = dart::math::Random::uniform(-0.2, 0.2);

	// this->mTargetPositions[3] += root_x_diff;
	// this->mTargetPositions[5] += root_z_diff;
	// this->mTargetPositions.segment<3>(0) = QuaternionToDARTPosition(Eigen::AngleAxisd(root_angle_diff, Eigen::Vector3d::UnitY())*DARTPositionToQuaternion(this->mTargetPositions.segment<3>(0)));

	this->mModifiedTargetPositions = this->mTargetPositions;
	auto& skel = this->mHumanoid->GetSkeleton();
	skel->setPositions(this->mTargetPositions);
	skel->setVelocities(this->mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;

	this->mVelRecords.clear();
	this->mRefRecords.clear();
	this->mModRecords.clear();
	this->mTimeRecords.clear();
	this->mGoalRecords.clear();
	this->mFootRecords.clear();
	this->mRefFootRecords.clear();
//	this->mBallRecords.clear();


	this->mTimeElapsed += 1.0 / this->mControlHz;
	this->mControlCount++;
	this->mUpdated = false;
}


void HumanoidController::ResetWithTime(double time_ratio)
{
	DPhy::Controller::Reset(false);

	//RSI
#if (USE_RNN == 1)
	if(this->mUseTrajectory){
		this->mTimeElapsed = this->GetMaxTime()*time_ratio;
		this->mControlCount = std::floor(this->mTimeElapsed*this->mControlHz);
		this->mStartTime = this->mTimeElapsed;
	}
	else{
		this->mTimeElapsed = 0.0;
		this->mControlCount = 0;
		this->mStartTime = this->mTimeElapsed;
	}
#elif (USE_RNN == 0)
		this->mTimeElapsed = (this->mBVH->GetMaxTime() - 0.51 - 1.0/this->mControlHz)*time_ratio;
		this->mControlCount = std::floor(this->mTimeElapsed*this->mControlHz);
		this->mStartTime = this->mTimeElapsed;
#endif

	this->mUpdated = false;
	this->UpdateReferenceDataForCurrentTime();
	// change root position and orientation
	// double root_x_diff = dart::math::Random::uniform(-0.1, 0.1);
	// double root_z_diff = dart::math::Random::uniform(-0.1, 0.1);
	// double root_angle_diff = dart::math::Random::uniform(-0.2, 0.2);

	// this->mTargetPositions[3] += root_x_diff;
	// this->mTargetPositions[5] += root_z_diff;
	// this->mTargetPositions.segment<3>(0) = QuaternionToDARTPosition(Eigen::AngleAxisd(root_angle_diff, Eigen::Vector3d::UnitY())*DARTPositionToQuaternion(this->mTargetPositions.segment<3>(0)));

	this->mModifiedTargetPositions = this->mTargetPositions;
	auto& skel = this->mHumanoid->GetSkeleton();
	skel->setPositions(this->mTargetPositions);
	skel->setVelocities(this->mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;

	this->mVelRecords.clear();
	this->mRefRecords.clear();
	this->mModRecords.clear();
	this->mTimeRecords.clear();
	this->mGoalRecords.clear();
	this->mFootRecords.clear();
	this->mRefFootRecords.clear();
//    this->mBallRecords.clear();

	this->mTimeElapsed += 1.0 / this->mControlHz;
	this->mControlCount++;
	this->mUpdated = false;
}


double HumanoidController::GetCurrentYRotation(){
	// from body joint vector
	Eigen::Vector3d ori = mHumanoid->GetSkeleton()->getPositions().segment<3>(0);
	Eigen::Quaterniond q(Eigen::AngleAxisd(ori.norm(), ori.normalized()));

	Eigen::Vector3d rotated = q._transformVector(Eigen::Vector3d::UnitZ());
	double angle = atan2(rotated[0], rotated[2]);

	return angle;
}

double HumanoidController::GetCurrentDirection(){
	// from com linear vel
	Eigen::Vector3d ori = mHumanoid->GetSkeleton()->getCOMLinearVelocity();
	double angle = atan2(ori[0], ori[2]);

	return angle;
}

void HumanoidController::ComputeRootCOMDiff(){
	auto& skel = mHumanoid->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::Vector3d com_diff;
	this->mRootCOMAtTerminal = skel->getCOM();
	skel->setPositions(mTargetPositions);
	skel->computeForwardKinematics(true,false,false);
	this->mRootCOMAtTerminalRef = skel->getCOM();

	skel->setPositions(p_save);
	skel->computeForwardKinematics(true,false,false);
}

void HumanoidController::GetRootCOMDiff(){
	if(mIsNanAtTerminal){
		std::cout << "Terminated by Nan value" << std::endl;
		return;
	} 
	printf("(% 11.6f, % 11.6f, % 11.6f), (% 11.6f, % 11.6f, % 11.6f)\n", 
		mRootCOMAtTerminal[0], mRootCOMAtTerminal[1], mRootCOMAtTerminal[2], 
		mRootCOMAtTerminalRef[0], mRootCOMAtTerminalRef[1], mRootCOMAtTerminalRef[2]
	);
}

bool
HumanoidController::
IsTerminalState()
{
	if(mIsTerminal)
		return true;

	mIsNanAtTerminal = false;

	auto& skel = mHumanoid->GetSkeleton();

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
	if(mUseTerminal){
		if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
			// std::cout << "root fall" << std::endl;
			mIsNanAtTerminal = false;
			// this->ComputeRootCOMDiff();
			mIsTerminal = true;
			terminationReason = 1;
		}
		if(std::abs(root_pos[0]) > 4990){
			mIsNanAtTerminal = false;
			// this->ComputeRootCOMDiff();
			mIsTerminal = true;
			terminationReason = 9;
		}
		if(std::abs(root_pos[2]) > 4990){
			mIsNanAtTerminal = false;
			// this->ComputeRootCOMDiff();
			mIsTerminal = true;
			terminationReason = 9;
		}
		 if(root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		 	mIsNanAtTerminal = false;
		 	// this->ComputeRootCOMDiff();
		 	mIsTerminal = true;
		 	terminationReason = 2;
		 }
		if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
			mIsNanAtTerminal = false;
			mIsTerminal = true;
			terminationReason = 5;
		}
	// p.segment<3>(3) = Eigen::Vector3d::Zero();
	// auto p_i_m = maxCoeff(p.cwiseAbs());
	// if(p_i_m.second>10){
	// 	// std::cout << "Thread " << omp_get_thread_num() << " p[" << p_i_m.first << "] exceed limit : " << p_i_m.second << std::endl;
	// 	mIsNanAtTerminal = true;
	// 	mIsTerminal = true;
	// 	terminationReason = 5;
	// }
	// v.segment<3>(3) = Eigen::Vector3d::Zero();
	// auto v_i_m = maxCoeff(v.cwiseAbs());
	// if(v_i_m.second>100){
	// 	// std::cout << "Thread " << omp_get_thread_num() << " v[" << v_i_m.first << "] exceed limit : " << v_i_m.second << std::endl;
	// 	mIsNanAtTerminal = true;
	// 	mIsTerminal = true;
	// 	terminationReason = 6;
	// }
	// if(mTimeElapsed>=mBVH->GetMaxTime()*1.0-1.0/mControlHz){
	}
#if(USE_RNN == 1)
	// if((!this->mUseTrajectory)&&mTimeElapsed>=1000.0){
	// 	// std::cout << "time end" << std::endl;
	// 	mIsNanAtTerminal = false;
	// 	this->ComputeRootCOMDiff();
	// 	mIsTerminal = true;
	// 	terminationReason = 8;
	// }
	if(this->mUseTrajectory){
		if(mUseDiscreteReference){
			if(this->mControlCount > this->GetMaxCount()){
				// std::cout << "time end" << std::endl;
				mIsNanAtTerminal = false;
				// this->ComputeRootCOMDiff();
				mIsTerminal = true;
				terminationReason =  8;
			}
		}
		else{
			if(this->mTimeElapsed > this->GetMaxTime()){
				// std::cout << "time end" << std::endl;
				mIsNanAtTerminal = false;
				// this->ComputeRootCOMDiff();
				mIsTerminal = true;
				terminationReason =  8;
			}
		}
	}
#elif(USE_RNN == 0)
	if(this->mTimeElapsed > this->mBVH->GetMaxTime()-0.51-1.0/this->mControlHz){
		// std::cout << "time end" << std::endl;
		mIsNanAtTerminal = false;
		// this->ComputeRootCOMDiff();
		mIsTerminal = true;
		terminationReason =  8;
	}
#endif
	return mIsTerminal;
}

void
HumanoidController::
SetGoalTrajectory(const std::vector<Eigen::Vector3d>& goal_trajectory)
{
	this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->setGoalTrajectory(goal_trajectory);
	// this->mReferenceManager->saveGoalTrajectory("goal_trajectory.txt");
}

void
HumanoidController::
SaveGoalTrajectory()
{
    this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->saveGoalTrajectory("goal_trajectory.txt");
}


Eigen::VectorXd
HumanoidController::
GetTargetPositions()
{
	return mTargetPositions;
}

Eigen::VectorXd
HumanoidController::
GetFuturePositions(double time, int index)
{
	if(index == -1)
		index = this->mCurrentReferenceManagerIndex;
	if(this->mUseTrajectory)
		return this->mReferenceManagers[index]->getPositions(this->mTimeElapsed + time);
	else
		return this->mReferenceManagers[index]->getPositions(time);
}

Eigen::VectorXd
HumanoidController::
GetFuturePositions(int count, int index)
{
	if(index == -1)
		index = this->mCurrentReferenceManagerIndex;
	if(this->mUseTrajectory)
		return this->mReferenceManagers[index]->getPositions(this->mControlCount + count);
	else
		return this->mReferenceManagers[index]->getPositions(count);
}

void
HumanoidController::
ApplyForce(std::string bodyname, Eigen::Vector3d force){
	this->mHumanoid->GetSkeleton()->getBodyNode(bodyname)->addExtForce(force);
}

Eigen::VectorXd
HumanoidController::
GetDecomposedPositions()
{
#ifdef NEW_JOINTS
	auto& skel = this->mHumanoid->GetSkeleton();
	Eigen::VectorXd pos = skel->getPositions();
	Eigen::VectorXd tp = pos;
	Eigen::Vector4d decomposed_root = rootDecomposition(this->mHumanoid->GetSkeleton(), tp);
	Eigen::VectorXd decomposed_positions(OUTPUT_MOTION_SIZE);

	skel->setPositions(tp);
	decomposed_positions.setZero();
	decomposed_positions[0] = tp[5]*100;
	decomposed_positions[1] = (tp[4]-ROOT_HEIGHT_OFFSET)*100;
	decomposed_positions[2] = -tp[3]*100;
	decomposed_positions.segment<4>(3) = decomposed_root;
	decomposed_positions.segment<36>(7) = tp.segment<36>(6);
	decomposed_positions.segment<9>(43) = tp.segment<9>(45);

	double foot_r_contact = 0;
	double foot_l_contact = 0;
	if( this->CheckCollisionWithGround("FootR")){
		foot_r_contact = 1;
	}
	if( this->CheckCollisionWithGround("FootEndR")){
		foot_r_contact = 1;
	}
	if( this->CheckCollisionWithGround("FootL")){
		foot_l_contact = 1;
	}
	if( this->CheckCollisionWithGround("FootEndL")){
		foot_l_contact = 1;
	}
	decomposed_positions[52] = foot_l_contact;
	decomposed_positions[53] = foot_r_contact;

	Eigen::Isometry3d cur_root_inv;
	cur_root_inv.setIdentity();
	cur_root_inv.linear() = Eigen::AngleAxisd(decomposed_root[0], Eigen::Vector3d::UnitY()).toRotationMatrix();
	cur_root_inv.translation() = Eigen::Vector3d(tp[3], 0.0, tp[5]);
	cur_root_inv = cur_root_inv.inverse();

	decomposed_positions.segment<3>(54) = changeToRNNPos(cur_root_inv * skel->getBodyNode("Head")->getTransform()*Eigen::Vector3d(0.0, 0.1, 0.0));

	decomposed_positions.segment<3>(57) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "HandL").translation());
	decomposed_positions.segment<3>(60) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "FootL").translation());

	decomposed_positions.segment<3>(63) = changeToRNNPos(cur_root_inv * skel->getBodyNode("FootL")->getTransform()*Eigen::Vector3d(0.0, 0.0, 0.15));

	decomposed_positions.segment<3>(66) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "HandR").translation());
	decomposed_positions.segment<3>(69) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "FootR").translation());

	decomposed_positions.segment<3>(72) = changeToRNNPos(cur_root_inv * skel->getBodyNode("FootR")->getTransform()*Eigen::Vector3d(0.0, 0.0, 0.15));

	decomposed_positions.segment<3>(75) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "ArmL").translation());
	decomposed_positions.segment<3>(78) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "ArmR").translation());

	decomposed_positions.segment<3>(81) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "ForeArmL").translation());
	decomposed_positions.segment<3>(84) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "TibiaL").translation());

	decomposed_positions.segment<3>(87) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "ForeArmR").translation());
	decomposed_positions.segment<3>(90) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "TibiaR").translation());

	decomposed_positions.segment<3>(93) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "Spine").translation());
	decomposed_positions.segment<3>(96) = changeToRNNPos(cur_root_inv * skel->getBodyNode("HandL")->getTransform()*Eigen::Vector3d(0.1, 0.0, 0.0));
	decomposed_positions.segment<3>(99) = changeToRNNPos(cur_root_inv * skel->getBodyNode("HandR")->getTransform()*Eigen::Vector3d(-0.15, 0.0, 0.0));
	decomposed_positions.segment<3>(102) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "Neck").translation());
	decomposed_positions.segment<3>(105) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "FemurL").translation());
	decomposed_positions.segment<3>(108) = changeToRNNPos(cur_root_inv * getJointTransform(skel, "FemurR").translation());
	skel->setPositions(pos);
	return decomposed_positions;
#else
	std::cout << "HumanoidController.cpp : GetDecomposedPositions is called without NEW_JOINTS!" << std::endl;
	exit(0);
#endif

}

int
HumanoidController::
GetNumState()
{
	return this->mNumState;
}
int
HumanoidController::
GetNumAction()
{
	return this->mNumAction;
}

Eigen::VectorXd
HumanoidController::
GetPositionState(const Eigen::VectorXd& pos){
	Eigen::VectorXd ret;

	auto& skel = mHumanoid->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();
	int num_body_nodes = mInterestedBodies.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	ret.resize(pos.rows()-6+6);
	ret.head(pos.rows()-6) = pos.tail(pos.rows()-6);

	skel->setPositions(pos);
	skel->computeForwardKinematics(true, false, false);

	// ret.resize((num_body_nodes)*6+9);
	// for(int i=0;i<num_body_nodes;i++)
	// {
	// 	// Eigen::Isometry3d transform = skel->getBodyNode(mInterestedBodies[i])->getTransform(root, root);
	// 	// Eigen::Quaterniond q = Eigen::Quaterniond(transform.linear());
	// 	// QuaternionNormalize(q);
	// 	int idx = skel->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
	// 	Eigen::Matrix3d rot = dart::dynamics::BallJoint::convertToRotation(pos.segment<3>(idx));
	// 	Eigen::VectorXd pos6(6);
	// 	pos6 << rot(0,1), rot(0,2), rot(1,0), rot(1,2), rot(2,0), rot(2,1);
	// 	ret.segment<6>(6*i) = pos6;
	// }

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	// Eigen::Matrix3d rot = transform.linear();
	// ret.tail<8>() << rot(0,0), rot(0,1), rot(0,2), rot(1,1), rot(1,2), transform.translation();
	Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
	ret.tail<6>() << rot, transform.translation();

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true, true, false);

	return ret;
}

Eigen::VectorXd 
HumanoidController::
GetEndEffectorState(const Eigen::VectorXd& pos){
	Eigen::VectorXd ret;

	auto& skel = mHumanoid->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	// int num_body_nodes = mInterestedBodies.size();
	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(pos);
	skel->computeForwardKinematics(true, false, false);

	ret.resize((num_ee)*6+6);
	for(int i=0;i<num_ee;i++)
	{
		// Eigen::Isometry3d transform = skel->getBodyNode(mInterestedBodies[i])->getTransform(root, root);
		// Eigen::Quaterniond q = Eigen::Quaterniond(transform.linear());
		// QuaternionNormalize(q);
		// int idx = skel->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);

		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		// Eigen::Matrix3d rot = transform.linear();
		// ret.segment<8>(8*i) << rot(0,0), rot(0,1), rot(0,2), rot(1,1), rot(1,2), transform.translation();
		Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
		ret.segment<6>(6*i) << rot, transform.translation();
	}
	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	// Eigen::Matrix3d rot = transform.linear();
	// ret.tail<8>() << rot(0,0), rot(0,1), rot(0,2), rot(1,1), rot(1,2), transform.translation();
	Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
	ret.tail<6>() << rot, transform.translation();
	// std::cout << transform.linear() << std::endl;
	// std::cout << transform.translation() << std::endl;
	// std::cout << ret.transpose() << std::endl;
	// std::cout << std::endl << std::endl;

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true, true, false);

	// Eigen::Vector3d root_diff;
	// root_diff = root->getTransform().linear().inverse()*(pos.segment<3>(3) - skel->getPositions().segment<3>(3));

	// Eigen::Quaterniond root_ori_diff;
	// root_ori_diff = DARTPositionToQuaternion(skel->getPositions().segment<3>(0)).inverse()
	// 				* DARTPositionToQuaternion(pos.segment<3>(0));		
	// QuaternionNormalize(root_ori_diff);
	// Eigen::Matrix3d rot = root_ori_diff.toRotationMatrix();
	// Eigen::VectorXd pos6(6);
	// pos6 << rot(0,1), rot(0,2), rot(1,0), rot(1,2), rot(2,0), rot(2,1);

	// ret.tail<9>() << root_diff, pos6;

	return ret;
}

Eigen::VectorXd HumanoidController::GetEndEffectorStatePosAndVel(const Eigen::VectorXd &pv) {
	Eigen::VectorXd ret;

	auto& skel = mHumanoid->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	// int num_body_nodes = mInterestedBodies.size();
	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	int size = pv.size();
	Eigen::VectorXd pos = pv.head(size/2);
	Eigen::VectorXd vel = pv.tail(size/2);

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

	// Eigen::Vector3d root_diff;
	// root_diff = root->getTransform().linear().inverse()*(pos.segment<3>(3) - skel->getPositions().segment<3>(3));

	// Eigen::Quaterniond root_ori_diff;
	// root_ori_diff = DARTPositionToQuaternion(skel->getPositions().segment<3>(0)).inverse()
	// 				* DARTPositionToQuaternion(pos.segment<3>(0));
	// QuaternionNormalize(root_ori_diff);
	// Eigen::Matrix3d rot = root_ori_diff.toRotationMatrix();
	// Eigen::VectorXd pos6(6);
	// pos6 << rot(0,1), rot(0,2), rot(1,0), rot(1,2), rot(2,0), rot(2,1);

	// ret.tail<9>() << root_diff, pos6;

	return ret;
}


bool
HumanoidController::
CheckCollisionWithGround(std::string bodyName){
	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	// auto collisionEngine = this->mGroundCollisionChecker;
	// auto cg1 = collisionEngine->createCollisionGroup(this->mWorld->getSkeleton("Ground").get());
	// auto cg2 = collisionEngine->createCollisionGroup(mHumanoid->GetSkeleton()->getBodyNode(bodyName));
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

void
HumanoidController::
UpdateInitialState(Eigen::VectorXd mod)
{
	auto& skel = mHumanoid->GetSkeleton();
	int nDof = skel->getNumDofs();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	double p_multiplier = 0.1;
	double v_multiplier = 0.2;

	// update positions
	for(int i = 0; i < nDof; i++){
		mod[i] = dart::math::clip(mod[i]*p_multiplier, -1.0, 1.0);
		p_save[i] += mod[i];
	}
	for(int i = nDof; i < nDof*2; i++){
		mod[i] *= v_multiplier;
		v_save[i-nDof] += mod[i];
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true, true, false);
}

Eigen::VectorXd
HumanoidController::
GetState()
{
	if(mIsTerminal)
		return Eigen::VectorXd::Zero(this->mNumState);
	this->UpdateReferenceDataForCurrentTime();
	auto& skel = mHumanoid->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	int num_body_nodes = mInterestedBodies.size();

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	/**********************************************************/
	// target angle and root diff
#if(USE_RNN == 1)
	// Eigen::Vector2d foot0 = this->mReferenceManager->getFootContacts(this->mTimeElapsed+1.0/this->mControlHz);
	// Eigen::Vector2d foot1 = this->mReferenceManager->getFootContacts(this->mTimeElapsed+0.2);
	// Eigen::Vector2d foot2 = this->mReferenceManager->getFootContacts(this->mTimeElapsed+0.4);
	// Eigen::Vector2d foot3 = this->mReferenceManager->getFootContacts(this->mTimeElapsed+0.6);
	// Eigen::Vector2d foot4 = this->mReferenceManager->getFootContacts(this->mTimeElapsed+0.8);
	// Eigen::Vector2d foot5 = this->mReferenceManager->getFootContacts(this->mTimeElapsed+1.0);

	// setting future pos times
	std::vector<Eigen::VectorXd> tp_vec;
	tp_vec.clear();
	if(this->mUseTrajectory){
		if(this->mUseDiscreteReference){
			std::vector<int> tp_times;
			tp_times.clear();
			tp_times.push_back(0);
			for(auto dt : tp_times){
				int t = std::max(0, this->mControlCount + dt);
				tp_vec.push_back(GetEndEffectorStatePosAndVel(this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getPositionsAndVelocities(t)));
			}
		}
		else{
			std::vector<double> tp_times;
			tp_times.clear();
			tp_times.push_back(0.05-1.0/this->mControlHz);
			for(auto dt : tp_times){
				double t = std::max(0.0, this->mTimeElapsed + dt);
				tp_vec.push_back(GetEndEffectorStatePosAndVel(this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getPositionsAndVelocities(t)));
			}
		}
	}
	else{
		if(this->mUseDiscreteReference){
			std::vector<int> tp_times;
			tp_times.clear();
			tp_times.push_back(0);
			for(auto dt : tp_times){
				int t = dt;
				tp_vec.push_back(GetEndEffectorStatePosAndVel(this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getPositionsAndVelocities(t)));
			}
		}
		else{
			std::vector<double> tp_times;
			tp_times.clear();
			tp_times.push_back(0.05-1.0/this->mControlHz);
			for(auto dt : tp_times){
				double t = dt;
				tp_vec.push_back(GetEndEffectorStatePosAndVel(this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getPositionsAndVelocities(t)));
			}
		}
	}
	Eigen::VectorXd tp_concatenated;
	tp_concatenated.resize(tp_vec.size()*tp_vec[0].rows());
	for(int i = 0; i < tp_vec.size(); i++){
		tp_concatenated.segment(i*tp_vec[0].rows(), tp_vec[0].rows()) = tp_vec[i];
	}
	// Eigen::VectorXd tp1 = GetEndEffectorState(this->mReferenceManager->getPositions(this->mTimeElapsed+0.2));
	// Eigen::VectorXd tp2 = GetEndEffectorState(this->mReferenceManager->getPositions(this->mTimeElapsed+0.4));
	// Eigen::VectorXd tp3 = GetEndEffectorState(this->mReferenceManager->getPositions(this->mTimeElapsed+0.6));
	// Eigen::VectorXd tp4 = GetEndEffectorState(this->mReferenceManager->getPositions(this->mTimeElapsed+0.8));
	// Eigen::VectorXd tp5 = GetEndEffectorState(this->mReferenceManager->getPositions(this->mTimeElapsed+1.0));
#elif(USE_RNN == 0)
	Eigen::VectorXd tp1 = GetPositionState(this->mHumanoid->GetTargetPositions(this->mBVH, this->mTimeElapsed+0.1));
	Eigen::VectorXd tp2 = GetPositionState(this->mHumanoid->GetTargetPositions(this->mBVH, this->mTimeElapsed+0.2));
	Eigen::VectorXd tp3 = GetPositionState(this->mHumanoid->GetTargetPositions(this->mBVH, this->mTimeElapsed+0.3));
	Eigen::VectorXd tp4 = GetPositionState(this->mHumanoid->GetTargetPositions(this->mBVH, this->mTimeElapsed+0.4));
	Eigen::VectorXd tp5 = GetPositionState(this->mHumanoid->GetTargetPositions(this->mBVH, this->mTimeElapsed+0.5));
#endif

	/**********************************************************/
	// angle and angular velocities of joints
	// com and linear velocities of body nodes
	Eigen::VectorXd p,v,pv;
	p.resize(p_save.rows()-6);
	p = p_save.tail(p_save.rows()-6);
	v = v_save/10.0;

	Eigen::VectorXd pv_tmp;
	pv_tmp.resize(p_save.rows()*2);
	pv_tmp << p_save, v_save;
	pv_tmp = GetEndEffectorStatePosAndVel(pv_tmp);
	pv.resize(pv_tmp.rows()-6);
	pv = pv_tmp.head(pv_tmp.rows()-6);
	pv.tail<6>() = pv_tmp.tail<6>();
	// v = v_save;

	// p.resize((num_body_nodes)*6);
	// v.resize((num_body_nodes)*3+6);
	// for(int i=0;i<num_body_nodes;i++)
	// {
	// 	int idx = skel->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
	// 	Eigen::Matrix3d rot = dart::dynamics::BallJoint::convertToRotation(p_save.segment<3>(idx));
	// 	Eigen::VectorXd pos6(6);
	// 	pos6 << rot(0,1), rot(0,2), rot(1,0), rot(1,2), rot(2,0), rot(2,1);
	// 	p.segment<6>(6*i) = pos6;

	// 	Eigen::Vector3d vel = v_save.segment<3>(idx);
	// 	v.segment<3>(3*i) = vel;
	// }
	// v.tail<6>() = root->getSpatialVelocity(dart::dynamics::Frame::World(), root);

	// root up vector
	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	// std::cout << up_vec.transpose() << ", " << up_vec_angle << std::endl;

	// chekc foot collision

	Eigen::VectorXd state;

	// dart::collision::CollisionResult cr = this->mWorld->getConstraintSolver()->getLastCollisionResult();
	// int fl2 = 0;
	// int fr2 = 0;
	// int num_contancts = cr.getNumContacts(); 

	// Eigen::Vector3d foot_left_contact_force = Eigen::Vector3d::Zero();
	// Eigen::Vector3d foot_right_contact_force = Eigen::Vector3d::Zero();
	// for(int i = 0; i < num_contancts; i++){
	// 	auto contact = cr.getContact(i);
	// 	auto collisionFrame1 = contact.collisionObject1->getShapeFrame();
	// 	const dart::dynamics::BodyNode *bn1, *bn2;
	// 	if( collisionFrame1->isShapeNode()){
	// 		bn1= collisionFrame1->asShapeNode()->getBodyNodePtr();
	// 	}

	// 	auto collisionFrame2 = contact.collisionObject2->getShapeFrame();
	// 	if( collisionFrame2->isShapeNode()){
	// 		bn2 = collisionFrame2->asShapeNode()->getBodyNodePtr();
	// 	}
	// 	if(bn1 == skel->getBodyNode("FootL")){
	// 		fl2 += 1;
	// 		foot_left_contact_force += contact.force;
	// 	}
	// 	if(bn1 == skel->getBodyNode("FootR")){
	// 		fr2 += 1;
	// 		foot_right_contact_force += contact.force;
	// 	}

	// 	if(bn2 == skel->getBodyNode("FootL")){
	// 		fl2 += 1;
	// 		foot_left_contact_force += -contact.force;
	// 	}
	// 	if(bn2 == skel->getBodyNode("FootR")){
	// 		fr2 += 1;
	// 		foot_right_contact_force -= contact.force;
	// 	}

	// }

	// foot_left_contact_force = skel->getRootBodyNode()->getTransform().linear().transpose()*foot_left_contact_force;
	// foot_right_contact_force = skel->getRootBodyNode()->getTransform().linear().transpose()*foot_right_contact_force;
	// foot_left_contact_force /= 1000;
	// foot_right_contact_force /= 1000;

	// foot height
	// const dart::dynamics::BodyNode *bn1, *bn2;
	// bn1 = skel->getBodyNode("FootL");
	// bn2 = skel->getBodyNode("FootR");

	// Eigen::Vector3d p0(0.0375, -0.1, 0.025);
	// Eigen::Vector3d p1(-0.0375, -0.1, 0.025);
	// Eigen::Vector3d p2(0.0375, 0.1, 0.025);
	// Eigen::Vector3d p3(-0.0375, 0.1, 0.025);

	// Eigen::Vector3d p0_l = bn1->getWorldTransform()*p0;
	// Eigen::Vector3d p1_l = bn1->getWorldTransform()*p1;
	// Eigen::Vector3d p2_l = bn1->getWorldTransform()*p2;
	// Eigen::Vector3d p3_l = bn1->getWorldTransform()*p3;

	// Eigen::Vector3d p0_r = bn2->getWorldTransform()*p0;
	// Eigen::Vector3d p1_r = bn2->getWorldTransform()*p1;
	// Eigen::Vector3d p2_r = bn2->getWorldTransform()*p2;
	// Eigen::Vector3d p3_r = bn2->getWorldTransform()*p3;

	// Eigen::VectorXd foot_corner_heights;
	// foot_corner_heights.resize(8);
	// foot_corner_heights << p0_l[1], p1_l[1], p2_l[1], p3_l[1], 
	// 						p0_r[1], p1_r[1], p2_r[1], p3_r[1];
	// foot_corner_heights *= 10;


	// root height
	double root_height = skel->getRootBodyNode()->getCOM()[1];
	double foot_l_height = skel->getBodyNode("FootL")->getCOM()[1];
	double foot_r_height = skel->getBodyNode("FootR")->getCOM()[1];
	double foot_end_l_height = skel->getBodyNode("FootEndL")->getCOM()[1];
	double foot_end_r_height = skel->getBodyNode("FootEndR")->getCOM()[1];

	double foot_r_contact = -1;
	double foot_end_r_contact = -1;
	double foot_l_contact = -1;
	double foot_end_l_contact = -1;
	if( this->CheckCollisionWithGround("FootR")){
		foot_r_contact = 1;
	}
	if( this->CheckCollisionWithGround("FootEndR")){
		foot_end_r_contact = 1;
	}
	if( this->CheckCollisionWithGround("FootL")){
		foot_l_contact = 1;
	}
	if( this->CheckCollisionWithGround("FootEndL")){
		foot_end_l_contact = 1;
	}

	double phase = 0;
	if(this->mUseTrajectory){
		if(this->mUseDiscreteReference){
			phase = (double)(this->mControlCount) / this->GetMaxCount();

			Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
			Eigen::Vector3d goal = cur_root_inv*this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getGoal(this->mControlCount);

		}
		else{
			phase = this->mTimeElapsed / this->GetMaxTime();

			Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
			Eigen::Vector3d goal = cur_root_inv*this->mReferenceManagers[this->mCurrentReferenceManagerIndex]->getGoal(this->mTimeElapsed);
		}
	}
#if (USE_RNN == 1)
	state.resize(p.rows()+v.rows()+tp_concatenated.rows()+1+5+6);
	state<<p, v, tp_concatenated, up_vec_angle,
			root_height, foot_l_height, foot_r_height, foot_end_l_height, foot_end_r_height,
			this->mRefFoot[0], this->mRefFoot[1], foot_l_contact, foot_end_l_contact, foot_r_contact, foot_end_r_contact;

#elif(USE_RNN == 0)
	int foot_l_collision = -1;
	int foot_r_collision = -1;
	if( this->CheckCollisionWithGround("FootL")){
		foot_l_collision = 1;
	}
	if( this->CheckCollisionWithGround("FootR")){
		foot_r_collision = 1;
	}

	state.resize(p.rows()+v.rows()+tp0.rows()+tp1.rows()+tp2.rows()+tp3.rows()+tp4.rows()+tp5.rows()+3+2*10);
	state<<p,v,tp1,tp2,tp3,tp4,tp5,up_vec,
			Eigen::VectorXd::Constant(10, foot_l_collision),Eigen::VectorXd::Constant(10, foot_r_collision)
			;
	// state.resize(p.rows()+v.rows()+tp0.rows()+3+2*10);
	// state<<p,v,tp0,up_vec,
	// 		Eigen::VectorXd::Constant(10, foot_l_collision),Eigen::VectorXd::Constant(10, foot_r_collision)
	// 		;
#endif
	// state.resize(p.rows()+v.rows()+tp.rows()+long_tp.rows()+3+2*10);
	// state<<p,v,tp,long_tp,up_vec,Eigen::VectorXd::Constant(10, foot_l_collision),Eigen::VectorXd::Constant(10, foot_r_collision);
	// state.resize(p.rows()+v.rows()+tp.rows()+3+20);
	// state<<p,v,tp,up_vec,Eigen::VectorXd::Constant(10, foot_l_collision),Eigen::VectorXd::Constant(10, foot_r_collision);

	return state;
}
void
HumanoidController::
SetAction(const Eigen::VectorXd& action)
{
	mActions = action;
}
double
HumanoidController::
GetReward()
{
	std::vector<double> ret = this->GetRewardByParts();
	return ret[0];
}

std::vector<double>
HumanoidController::
GetRewardByParts()
{
	auto& skel = this->mHumanoid->GetSkeleton();

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
		int idx = mHumanoid->GetSkeleton()->getBodyNode(mRewardBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		p_diff_lower.segment<3>(3*i) = p_diff.segment<3>(idx);
		v_diff_lower.segment<3>(3*i) = v_diff.segment<3>(idx);
	}

	//End-effector position and COM Differences
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	// Eigen::MatrixXd mass_diff = skel->getMassMatrix();

	std::vector<Eigen::Isometry3d> ee_transforms;
	Eigen::VectorXd ee_diff(mEndEffectors.size()*3);
	Eigen::VectorXd ee_ori_diff(mEndEffectors.size()*3);
	Eigen::Vector3d com_diff, com_v_diff;

	// Eigen::VectorXd p_global_diff(this->mRewardBodies.size()*3);
	// p_global_diff.setZero();


	for(int i=0;i<mEndEffectors.size();i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
	}
	// for(int i=0; i < this->mRewardBodies.size(); i++){
	// 	p_global_diff.segment<3>(3*i) = skel->getBodyNode(this->mRewardBodies[i])->getWorldTransform().translation();
	// }
	com_diff = skel->getCOM();
	com_v_diff = skel->getCOMLinearVelocity();


	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);
	// mass_diff -= skel->getMassMatrix();

	for(int i=0;i<mEndEffectors.size();i++){
		Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee_diff.segment<3>(3*i) = diff.translation();
		ee_ori_diff.segment<3>(3*i) = QuaternionToDARTPosition(Eigen::Quaterniond(diff.linear()));
	}
	// for(int i=0; i < this->mRewardBodies.size(); i++){
	// 	p_global_diff.segment<3>(3*i) -= skel->getBodyNode(this->mRewardBodies[i])->getWorldTransform().translation();
	// }
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



	// goal
	// Eigen::Vector3d local_goal = skel->getRootBodyNode()->getTransform().inverse() * this->mGoal;
	// local_goal[1] = 0;

	double scale = 1.0;
	double sig_p = 0.1 * scale; 		// 2
	double sig_v = 1.0 * scale;		// 3
	double sig_com = 0.3 * scale;		// 4
	double sig_ee = 0.3 * scale;		// 8

	double sig_com_v = 0.2 * scale;		// 5
	double sig_ori = 0.8 * scale;		// 6
	double sig_av = 4.0 * scale;		// 7
	double sig_ee_ori = 1.2 * scale;	// 9

	// double sig_mass = 2.0 * scale;

	// double sig_goal = 10.0 * scale;		// 
	// double sig_p_g = 2.5 * scale;		// 

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
	// r_tot = 0.0+0.9*r_tot + 0.05*(foot_contact_match_r+foot_contact_match_l);
	// double r_tot = (w_p*r_p 
	// 				+ w_v*r_v 
	// 				+ w_root_ori*r_ori 
	// 				+ w_root_av*r_av
	// 				+ w_ee*r_ee 
	// 				+ w_goal*r_goal)	
	// 				* r_com;
	// double r_tot = r_p * r_v * r_ori * r_av * r_ee * r_com * r_goal;
					// + w_goal*r_goal;
	// std::cout<<"-------Reward-------"<<std::endl;
	// std::cout<<"p  : "<<r_p<<std::endl;
	// std::cout<<"v  : "<<r_v<<std::endl;
	// std::cout<<"ee : "<<r_ee<<std::endl;
	// std::cout<<"com: "<<r_com<<std::endl;

	// std::cout<<"suM: "<<r_sum<<std::endl;
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
	// ret.push_back(r_com_v);
	// ret.push_back(r_ori);
	// ret.push_back(r_av);
	ret.push_back(r_ee);
	// ret.push_back(r_mass);
	return ret;
}

void HumanoidController::Record()
{
	int total_dof = 0;
	for(auto& ch : mCharacters){
		total_dof += ch.second->GetSkeleton()->getNumDofs();
	}

	Eigen::VectorXd record(total_dof);

	int cur_pos = 0;
	for(auto& ch : mCharacters){
		int cur_dof = ch.second->GetSkeleton()->getNumDofs();
		record.segment(cur_pos, cur_dof) = ch.second->GetSkeleton()->getPositions();

		cur_pos += cur_dof;
	}

	int foot_l_collision = -1;
	int foot_r_collision = -1;
	if( this->CheckCollisionWithGround("FootL")){
		foot_l_collision = 1;
	}
	if( this->CheckCollisionWithGround("FootR")){
		foot_r_collision = 1;
	}

	Eigen::Vector2d foot;
	foot << foot_l_collision, foot_r_collision;

	this->mRecords.push_back(record);
	this->mRefRecords.push_back(mTargetPositions);
	this->mModRecords.push_back(mModifiedTargetPositions);
	this->mVelRecords.push_back(this->mHumanoid->GetSkeleton()->getCOMLinearVelocity());
	this->mTimeRecords.push_back(this->mTimeElapsed);
	this->mGoalRecords.push_back(this->mGoal);
	this->mFootRecords.push_back(foot);
	this->mRefFootRecords.push_back(this->mRefFoot);

	if(useBall)
    {
	    int ballNum= mThrowingBall->mBalls.size();
	    Eigen::VectorXd ballRecord(1+ballNum*4);
	    ballRecord[0]= ballNum;

	    int cur_pos=1;
	    for(auto& ball: mThrowingBall->mBalls)
        {
//	        string ball_name= ball->getName();
//	        string ball_name_int_start= ball_name.find_first_of("0123456789");
//	        int ball_index= std::stoi(ball_name.substr( ball_name_int_start, ball_name.length()- ball_name_int_start));
            int ball_index= std::stoi(ball->getName().substr(5));
	        ballRecord[cur_pos]= ball_index;
	        ballRecord.segment(cur_pos+1, 3)= ball->getBodyNode(0)->getWorldTransform().translation();
//	        ballRecord.segment(cur_pos+4, 3)= ball->getVelocities().tail<3>();
	        cur_pos= cur_pos+4;
        }
	    this->mBallRecords.push_back(ballRecord);
//	    std::cout<<"bR: "<<ballNum<<" , "<<ballRecord.transpose()<<std::endl;
    }
}


void HumanoidController::WriteRecords(const std::string& filename){
	// std::cout << "Write Records" << std::endl;
	std::ofstream ofs(filename);
	// first line : number of character
	int nCharacters = this->mCharacters.size();
	ofs << nCharacters << std::endl;

	// next lines : file paths of characters;
	// for(int i = 0; i < nCharacters; i++){
	for(auto& ch : mCharacters){
		ofs << ch.second->GetCharacterPath() << std::endl;
	}

	// motion file name of character
	// need to fix when add another character not humanoid
	// ofs << this->mReferenceMotionFilename << std::endl;


	// next line : number of record size;
	int nFrames = this->mRecords.size();
	ofs << nFrames << std::endl;
	
	// next line : time step;
	double time_step = this->mWorld->getTimeStep();
	ofs << mStartTime << std::endl;
	ofs << time_step << std::endl;
	
	// next lines : Records data;
	
	for(int i=0;i<nFrames;i++){
			ofs << mRecords[i].transpose() << std::endl;
	}

	ofs << "Refs" << std::endl;
	for(int i=0;i<nFrames;i++){
			ofs << this->mRefRecords[i].transpose() << std::endl;
	}

	ofs << "Vels" << std::endl;
	for(int i=0;i<nFrames;i++){
			ofs << this->mVelRecords[i].transpose() << std::endl;
	}

	ofs << "Goals" << std::endl;
	for(int i=0;i<nFrames;i++){
			ofs << this->mGoalRecords[i].transpose() << std::endl;
	}

	ofs << "Mods" << std::endl;
	for(int i=0;i<nFrames;i++){
			ofs << this->mModRecords[i].transpose() << std::endl;
	}

	ofs << "Foots" << std::endl;
	for(int i=0;i<nFrames;i++){
			ofs << this->mFootRecords[i].transpose() << std::endl;
	}
	ofs << "RefFoots" << std::endl;
	for(int i=0;i<nFrames;i++){
			ofs << this->mRefFootRecords[i].transpose() << std::endl;
	}

	// ofs << "Times" << std::endl;
	// for(int i=0;i<nFrames;i++){
	// 		ofs << this->mTimeRecords[i] << std::endl;
	// }	

	if(useBall)
    {
        ofs << "Balls" << std::endl;
        for(int i=0;i<nFrames;i++){
            ofs << this->mBallRecords[i].transpose() << std::endl;
        }
    }

	ofs.close();

}
void HumanoidController::WriteCompactRecords(const std::string& filename)
{
	// std::cout << this->mMaxJoint.transpose() << std::endl;
	// std::cout << this->mMinJoint.transpose() << std::endl;
	// std::cout << "Write Records" << std::endl;
	std::ofstream ofs(filename);
	// first line : number of character
	int nCharacters = this->mCharacters.size();
	ofs << nCharacters << std::endl;

	// next lines : file paths of characters;
	for(auto& ch : mCharacters){
		ofs << ch.second->GetCharacterPath() << std::endl;
	}

	// motion file name of character
	// need to fix when add another character not humanoid
	// ofs << this->mReferenceMotionFilename << std::endl;

	double rendering_time_step = 1.0/30;
	int stride = rendering_time_step/this->mWorld->getTimeStep();

	if(stride == 0){
		stride = 1;
	}

	// next line : number of record size;
	int nFrames = std::ceil((double)this->mRecords.size()/stride);
	ofs << nFrames << std::endl;
	
	// next line : time step;
	double time_step = stride*this->mWorld->getTimeStep();
	ofs << mStartTime << std::endl;
	ofs << time_step << std::endl;
	
	// next lines : Records data;
	std::cout<<"Record Size : "<<nFrames<<", terminated by "<<terminationReason<<std::endl;
	// std::cout << this->mRecords.size() << ", " << stride << std::endl;
	
	for(int i=stride-1;i<this->mRecords.size();i+=stride){
			ofs << mRecords[i].transpose() << std::endl;
	}

	ofs << "Refs" << std::endl;
	for(int i=stride-1;i<this->mRefRecords.size();i+=stride){
			ofs << this->mRefRecords[i].transpose() << std::endl;
	}

	ofs << "Vels" << std::endl;
	for(int i=stride-1;i<this->mVelRecords.size();i+=stride){
			ofs << this->mVelRecords[i].transpose() << std::endl;
	}

	ofs << "Goals" << std::endl;
	for(int i=stride-1;i<this->mGoalRecords.size();i+=stride){
			ofs << this->mGoalRecords[i].transpose() << std::endl;
	}

	ofs << "Mods" << std::endl;
	for(int i=stride-1;i<this->mModRecords.size();i+=stride){
			ofs << this->mModRecords[i].transpose() << std::endl;
	}

	ofs << "Foots" << std::endl;
	for(int i=stride-1;i<this->mFootRecords.size();i+=stride){
			ofs << this->mFootRecords[i].transpose() << std::endl;
	}

	ofs << "RefFoots" << std::endl;
	for(int i=stride-1;i<this->mRefFootRecords.size();i+=stride){
			ofs << this->mRefFootRecords[i].transpose() << std::endl;
	}

    if(useBall)
    {
        ofs << "Balls" << std::endl;
        for(int i=stride-1;i<this->mBallRecords.size();i+=stride){
            ofs << this->mBallRecords[i].transpose() << std::endl;
        }
//        for(int i=0;i<nFrames;i++){
//            ofs << this->mBallRecords[i].transpose() << std::endl;
//        }
    }

	ofs.close();
}


}

