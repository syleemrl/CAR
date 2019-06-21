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

	/*this->mBVH = new BVH();
	std::string motionfilename = std::string(DPHY_DIR)+std::string("/motion/dribble_test.bvh");
	this->mBVH->Parse(motionfilename);
	this->mHumanoid->InitializeBVH(this->mBVH);*/

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
	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();
	torques.clear();
}
void Controller::Step()
{
	int per = mSimulationHz/mControlHz;
	if(IsTerminalState())
		return;
	
	// set action target pos
	int num_body_nodes = this->mInterestedBodies.size();
	double pd_gain_offset = 0;

	mModifiedTargetPositions = this->mTargetPositions;
	mModifiedTargetVelocities = this->mTargetVelocities;
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

	this->mTimeElapsed += 1.0 / this->mControlHz;
	this->mControlCount++;
	this->mUpdated = false;
}
void Controller::Reset(bool RSI)
{
	mFrame = 0;
	mWorld->reset();
	mRecords.clear();
	for(auto ch : mCharacters)
	{
		auto& skel = ch.second->GetSkeleton();
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
	}
}
