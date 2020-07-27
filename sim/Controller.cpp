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
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false),
	mRD(), mMT(mRD()), mDistribution(0.85, 1.25)
{
	this->sig_torque = 0.4;
	this->mRescaleParameter = std::make_tuple(1.0, 1.0, 1.0);
	this->isAdaptive = adaptive;
	this->mRecord = record;
	this->mReferenceManager = ref;
	this->id = id;

	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();
	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81,0));

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml")).first;
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
	
	mInterestedDof = 0;
	for(int i = 0; i < mInterestedBodies.size(); i++) {
		mInterestedDof += mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getNumDofs();
	}
	mRewardDof = 0;
	for(int i = 0; i < mRewardBodies.size(); i++) {
		mRewardDof += mCharacter->GetSkeleton()->getBodyNode(mRewardBodies[i])->getParentJoint()->getNumDofs();
	}
	mAdaptiveBodies.clear();
	mAdaptiveBodies.push_back("Torso");
	mAdaptiveBodies.push_back("Spine");
	mAdaptiveBodies.push_back("FemurR");
	mAdaptiveBodies.push_back("FemurL");	
	mAdaptiveBodies.push_back("TibiaR");
	mAdaptiveBodies.push_back("TibiaL");
	mAdaptiveBodies.push_back("ArmR");
	mAdaptiveBodies.push_back("ArmL");
	mAdaptiveBodies.push_back("ForeArmR");
	mAdaptiveBodies.push_back("ForeArmL");

	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	this->mCGL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootL"));
	this->mCGR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootR"));
	this->mCGEL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootEndL"));
	this->mCGER = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootEndR"));
	this->mCGHL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("HandL"));
	this->mCGHR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("HandR"));
	this->mCGG = collisionEngine->createCollisionGroup(this->mGround.get());

	int num_body_nodes = this->mInterestedBodies.size();
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	this->mMask.resize(dof);
	this->mMask.setZero();	
	mActions = Eigen::VectorXd::Zero(mInterestedDof + 1);
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

	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	this->mPDTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mPDTargetVelocities = Eigen::VectorXd::Zero(dof);

	mTarget = mDistribution(mMT);

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
	this->mRecordTargetPosition.clear();
	this->mRecordBVHPosition.clear();
	this->mRecordRewardPosition.clear();
	this->mRecordObjPosition.clear();

	mControlFlag.resize(2);
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
	mOpMode = false;

	if(mRecord) {
		path = std::string(CAR_DIR)+std::string("/character/sandbag.xml");
		this->mObject = new DPhy::Character(path);
		this->mCGOBJ = collisionEngine->createCollisionGroup(this->mObject->GetSkeleton()->getBodyNode("Sandbag"));
		this->mWorld->addSkeleton(this->mObject->GetSkeleton());

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
std::vector<double>
Controller::
GetAdaptiveIdxs()
{
	std::vector<double> idxs;
	idxs.clear();
	idxs.push_back(3);
	for(int i = 0; i < mAdaptiveBodies.size(); i++) {
		int idx = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		idxs.push_back(idx);
	}
	return idxs;
}
void 
Controller::
Step()
{
	if(IsTerminalState())
		return;

	// set action target pos
	int num_body_nodes = this->mInterestedBodies.size();
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	for(int i = 0; i < mInterestedDof; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.2, -0.7*M_PI, 0.7*M_PI);
	}

	mActions[mInterestedDof] = dart::math::clip(mActions[mInterestedDof]*0.05, -0.8, 0.8);
	mAdaptiveStep = mActions[mInterestedDof];
	// if(isAdaptive)
	// 	mAdaptiveStep *= 0.1;

	mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame += (1 + mAdaptiveStep);
	this->mCurrentFrameOnPhase += (1 + mAdaptiveStep);

	if(this->mCurrentFrameOnPhase > mReferenceManager->GetPhaseLength()){
		this->mCurrentFrameOnPhase -= mReferenceManager->GetPhaseLength();
		mHeadRoot = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		mControlFlag.setZero();
		if(isAdaptive) {
			mReferenceManager->SaveTrajectories(data_spline, std::pair<double, double>(mTrackingRewardTrajectory, mTargetRewardTrajectory));
			data_spline.clear();
			mTrackingRewardTrajectory = 0;
			mTargetRewardTrajectory = 0;
		}
	}
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame, isAdaptive);
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity() * (1 + mAdaptiveStep);
	
	delete p_v_target;

	p_v_target = mReferenceManager->GetMotion(mCurrentFrame, false);
	this->mPDTargetPositions = p_v_target->GetPosition();
	this->mPDTargetVelocities = p_v_target->GetVelocity() * (1 + mAdaptiveStep);
	
	delete p_v_target;

	// if(mOpMode) {
	// 	Motion* ref1 = mReferenceManager->GetMotionForOptimization(mCurrentFrame, id);
	// 	Motion* ref2 = mReferenceManager->GetMotion(mCurrentFrame, true);
		
	// 	Eigen::VectorXd p1 = ref1->GetPosition();
	// 	Eigen::VectorXd p2 = ref2->GetPosition();

	// 	Eigen::VectorXd d(dof);

	// 	for(int j = 0; j < n_bnodes; j++) {
	// 		int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
	// 		int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			
	// 		if(dof == 6) {
	// 			d.segment<3>(idx) = JointPositionDifferences(p1.segment<3>(idx), p2.segment<3>(idx));
	// 			d.segment<3>(idx + 3) = p1.segment<3>(idx + 3) -  p2.segment<3>(idx + 3);
	// 		} else if (dof == 3) {
	// 			d.segment<3>(idx) = JointPositionDifferences(p1.segment<3>(idx), p2.segment<3>(idx));
	// 		} else {
	// 			d(idx) = p1(idx) - p2(idx);
	// 		}
	// 	}

	// 	for(int j = 0; j < n_bnodes; j++) {
	// 		int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
	// 		int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
	// 		if(dof == 6) {
	// 			mPDTargetPositions.segment<3>(idx) = Rotate3dVector(mPDTargetPositions.segment<3>(idx), d.segment<3>(idx));
	// 			mPDTargetPositions.segment<3>(idx + 3) = d.segment<3>(idx + 3) + mPDTargetPositions.segment<3>(idx + 3);
	// 		} else if (dof == 3) {
	// 			mPDTargetPositions.segment<3>(idx) = Rotate3dVector(mPDTargetPositions.segment<3>(idx), d.segment<3>(idx));
	// 		} else {
	// 			mPDTargetPositions(idx) = d(idx) + mPDTargetPositions(idx);
	// 		}
	// 	}
	// 	this->mTargetPositions = ref1->GetPosition();
	// 	delete ref1, ref2;
	// } 

	int count_dof = 0;
	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		int dof = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getNumDofs();
		mPDTargetPositions.block(idx, 0, dof, 1) += mActions.block(count_dof, 0, dof, 1);
		count_dof += dof;

	}

	// set pd gain action
	Eigen::VectorXd kp(mCharacter->GetSkeleton()->getNumDofs()), kv(mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();

	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		int dof = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getNumDofs();

		if(mInterestedBodies[i] == "Spine"){
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

	double torque_sum = 0;
	double end_F_sum_norm = 0;
	Eigen::Vector3d end_F_sum;
	end_F_sum.setZero();
	for(int i = 0; i < this->mSimPerCon; i += 2){
		torque = mCharacter->GetSPDForces(mPDTargetPositions, mPDTargetVelocities);
		for(int j = 0; j < num_body_nodes; j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[j])->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[j])->getParentJoint()->getNumDofs();

			double torquelim = mCharacter->GetTorqueLimit(mInterestedBodies[j]);
			double torque_norm = torque.block(idx, 0, dof, 1).norm();
			torque.block(idx, 0, dof, 1) = std::max(-torquelim, std::min(torquelim, torque_norm)) * torque.block(idx, 0, dof, 1).normalized();
		}
		auto end_node = mCharacter->GetSkeleton()->getBodyNode("HandR");
		Eigen::MatrixXd J = mCharacter->GetSkeleton()->getLinearJacobian(mCharacter->GetSkeleton()->getBodyNode("HandR"), Eigen::Vector3d(0, 0, 0));
		Eigen::Vector3d end_F = J * torque;
		end_F_sum += 2.0 * end_F / mSimulationHz;
		end_F_sum_norm += 2.0 * end_F.norm() / mSimulationHz;
		for(int j = 0; j < 2; j++)
		{
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step();
		}
		mTimeElapsed += 2 * (1 + mAdaptiveStep);
	}

	nTotalSteps += 1;

	if(mCurrentFrameOnPhase >= 21 && mCurrentFrameOnPhase <= 27)  
		mRecordWork.push_back(end_F_sum_norm);
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
		Eigen::VectorXd p(mCharacter->GetSkeleton()->getPositions().rows() + 1);
		p << mCharacter->GetSkeleton()->getPositions(), mAdaptiveStep;
		data_spline.push_back(std::pair<Eigen::VectorXd,double>(p, mCurrentFrame));
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
	mRecordRewardPosition.push_back(mRewardTargetPositions);
	if(mOpMode)
		mRecordBVHPosition.push_back(mReferenceManager->GetPosition(mCurrentFrame, true));
	else
		mRecordBVHPosition.push_back(mReferenceManager->GetPosition(mCurrentFrame, false));
	
	mRecordObjPosition.push_back(mObject->GetSkeleton()->getPositions());
	mRecordTargetPosition.push_back(mTargetPositions);
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	mRecordCOM.push_back(mCharacter->GetSkeleton()->getCOM());
	mRecordTime.push_back(mCurrentFrame);
	
	bool rightContact = CheckCollisionWithGround("FootEndR") || CheckCollisionWithGround("FootR");
	bool leftContact = CheckCollisionWithGround("FootEndL") || CheckCollisionWithGround("FootL");

	mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));
}
std::vector<double> 
Controller::
GetTrackingReward(Eigen::VectorXd position, Eigen::VectorXd position2, 
	Eigen::VectorXd velocity, Eigen::VectorXd velocity2, std::vector<std::string> list, bool useVelocity)
{
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	Eigen::VectorXd p_diff = skel->getPositionDifferences(position, position2);
	Eigen::VectorXd p_diff_reward;
	
	p_diff_reward.resize(mRewardDof);
	int count_dof = 0;

	for(int i = 0; i < list.size(); i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(list[i])->getParentJoint()->getIndexInSkeleton(0);
		int dof = mCharacter->GetSkeleton()->getBodyNode(list[i])->getParentJoint()->getNumDofs();
		
		p_diff_reward.block(count_dof, 0, dof, 1) = p_diff.block(idx, 0, dof, 1);
		count_dof += dof;
	}
	Eigen::VectorXd v_diff, v_diff_reward;

	if(useVelocity) {
		v_diff = skel->getVelocityDifferences(velocity, velocity2);
		v_diff_reward.resize(mRewardDof);
		count_dof = 0;

		for(int i = 0; i < list.size(); i++){
			int idx = mCharacter->GetSkeleton()->getBodyNode(list[i])->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(list[i])->getParentJoint()->getNumDofs();

			v_diff_reward.block(count_dof, 0, dof, 1) = v_diff.block(idx, 0, dof, 1);
			count_dof += dof;
		}
	}

	skel->setPositions(position);
	if(useVelocity) skel->setVelocities(velocity);
	skel->computeForwardKinematics(true,useVelocity,false);

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

	double sig_p = 0.5 * scale; 
	double sig_v = 2 * scale;	
	double sig_com = 0.2 * scale;		
	double sig_ee = 0.1 * scale;		

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
// double
// Controller::
// GetSimilarityReward()
// {
// 	int dof = mCharacter->GetSkeleton()->getNumDofs();
// 	Eigen::VectorXd curPos = mCharacter->GetSkeleton()->getPositions();
// 	Eigen::AngleAxisd prev_root_ori = Eigen::AngleAxisd(mPrevPositions.segment<3>(0).norm(), mPrevPositions.segment<3>(0).normalized());
//  	Eigen::VectorXd diff_local(dof);
//  	int count = 0;
//  	for(int i = 0; i < mAdaptiveBodies.size(); i++) {
//  		int idx = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getIndexInSkeleton(0);
//  		int dof = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getNumDofs();

//  		if(dof == 6) {
//  			diff_local.segment<3>(count) = JointPositionDifferences(curPos.segment<3>(idx), mPrevPositions.segment<3>(idx));
//  			diff_local.segment<3>(count + 3) = curPos.segment<3>(idx + 3) - mPrevPositions.segment<3>(idx + 3);
//  			diff_local.segment<3>(count + 3) = prev_root_ori.inverse() * diff_local.segment<3>(count + 3);
//  		} else if(dof == 3) {
//  			diff_local.segment<3>(count) = JointPositionDifferences(curPos.segment<3>(idx), mPrevPositions.segment<3>(idx));
//  		} else {
//  			diff_local(count) = curPos(count) - mPrevPositions(count);
//  		}
//  		count += dof;
//  	}

//  	double x, y, a, b;

//  	Eigen::VectorXd mean = mReferenceManager->GetAxisMean(mPrevFrameOnPhase);
//  	Eigen::VectorXd dev = mReferenceManager->GetAxisDev(mPrevFrameOnPhase);
	
// 	double max = 10;
// 	double min = 0.5;

//  	count = 0;
//  	double diff = 0;

//  	for(int i = 0; i < mAdaptiveBodies.size(); i++) {
//  		int dof = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getNumDofs();
//  		int idx = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getIndexInSkeleton(0);
 		
//  		if(dof == 6) {
// 			Eigen::VectorXd v = diff_local.block(count, 0, 3, 1) - mean.segment<3>(idx);
// 	 		x = v.dot(mean.segment<3>(idx).normalized());
//  			y = (v - x * mean.segment<3>(idx).normalized()).norm() / std::max(mean.segment<3>(idx).norm(), 0.02);
//  			x /= std::max(mean.segment<3>(idx).norm(), 0.02);

// 			a = std::max(max - 0.07 * dev(idx) * (max-min), 1.0) + 1e-8;
// 			b = std::min(min + 0.02 * dev(idx) * (max-min), 1.5) + 1e-8;
// 			double d = (x * x) / (a * a) + (y * y) / (b * b);
// 			diff += exp(-d * 5);
// 			std::cout << mCurrentFrameOnPhase << " " <<mAdaptiveBodies[i] << " " << d << " " << exp(-d * 5) << std::endl;

// 			v = diff_local.block(count+3, 0, 3, 1) - mean.segment<3>(idx + 3);
// 	 		x = v.dot(mean.segment<3>(idx + 3).normalized());
//  			y = (v - x * mean.segment<3>(idx + 3).normalized()).norm() / std::max(mean.segment<3>(idx + 3).norm(), 0.02);
//  			x /= std::max(mean.segment<3>(idx + 3).norm(), 0.02);

//  			a = std::max(max - 0.07 * dev(idx + 3) * (max-min), 1.0) + 1e-8;
// 			b = std::min(min + 0.02 * dev(idx + 3) * (max-min), 1.5) + 1e-8;
			
// 			d = (x * x) / (a * a) + (y * y) / (b * b);
// 			diff += exp(-d * 5);
// 			std::cout << mCurrentFrameOnPhase << " " <<mAdaptiveBodies[i] << " " << d << " " << exp(-d * 5) << std::endl;
		
//  		} else {
// 			Eigen::VectorXd v = diff_local.block(count, 0, dof, 1) - mean.block(idx, 0, dof, 1);
// 	 		x = v.dot(mean.block(idx, 0, dof, 1).normalized());
//  			y = (v - x * mean.block(idx, 0, dof, 1).normalized()).norm() / std::max(mean.block(idx, 0, dof, 1).norm(), 0.02);
//  			x /= std::max(mean.block(idx, 0, dof, 1).norm(), 0.02);

// 			a = std::max(max - 0.07 * dev(idx) * (max-min), 1.0) + 1e-8;
// 			b = std::min(min + 0.02 * dev(idx) * (max-min), 1.5) + 1e-8;
		
// 			double d = (x * x) / (a * a) + (y * y) / (b * b);
// 			diff += exp(-d * 5);
			
// 			std::cout << mCurrentFrameOnPhase << " " <<mAdaptiveBodies[i] << " " << x << " " << y << " "<< a << " " << b << " " << d << " " << exp(-d * 5) << std::endl;
//  		}
//  		count += dof;
//  	}
//  	diff /= (mAdaptiveBodies.size() + 1);
//  	std::cout << diff << std::endl;
// 	return diff;
// }
double 
Controller::
GetTargetReward()
{
	double r_target = 0;
	auto& skel = this->mCharacter->GetSkeleton();

	//jump	
	// if(mCurrentFrameOnPhase >= 44 && mControlFlag[0] == 0) {
	// 	double target_diff = skel->getCOM()[1] - 1.45;
	// 	r_target = 2 * exp(-pow(target_diff, 2) * 30);
	// 	mControlFlag[0] = 1;
	// 	target_reward = skel->getCOM()[1];
	// 	meanTargetReward = meanTargetReward * (mCount / (mCount + 1.0)) + r_target * (1.0 / (mCount + 1.0));
	// 	mCount += 1;
	// 	if(mRecord)
	// 		std::cout << skel->getCOM()[1] << " " << r_target << std::endl;
	// }

	// if(mCurrentFrameOnPhase >= 35 && mCurrentFrameOnPhase < 39 && mControlFlag[0] == 0) {
	// 	mTarget = mRecordWork.back();
	// 	mTarget2 = 1;
	// 	mControlFlag[0] = 1;
	// } else if (mCurrentFrameOnPhase >= 39 && mCurrentFrameOnPhase < 44 && mControlFlag[0] == 1) {
	// 	mTarget = mTarget / mTarget2;
	// 	mControlFlag[0] = 0;
	// 	r_target = exp(-pow(mTarget - 1.2, 2)*0.5)*1.5;
	// 	std::cout << mTarget << std::endl;
	// 	// meanTargetReward = meanTargetReward * (mCount / (mCount + 1.0)) + r_target * (1.0 / (mCount + 1.0));
	// 	// mCount += 1;

	// //	std::cout << mTarget << " " << r_target << std::endl;
	// } if(mCurrentFrameOnPhase >= 35 && mCurrentFrameOnPhase < 39 && mControlFlag[0] == 1) {
	// 	mTarget += mRecordWork.back();
	// 	mTarget2 += 1;
	// }

	//jump turn	
	// if(mCurrentFrameOnPhase >= 36.0 && mControlFlag[0] == 0) {
	// 	mTarget = 0;
	// 	mControlFlag[0] = 1;
	// } else if(mCurrentFrameOnPhase >= 68.0 && mControlFlag[0] == 1) {
	// 	mControlFlag[0] = -1;
	// 	double target_diff = mTarget - 5.5;
	// 	r_target = 2*exp(-pow(target_diff, 2)*0.3);
	// 	mReferenceManager->SetTargetReward(mTarget, id);
	// 	meanTargetReward = meanTargetReward * (mCount / (mCount + 1.0)) + r_target * (1.0 / (mCount + 1.0));
	// 	mCount += 1;
	// //	std::cout << mTarget << " " << r_target << std::endl;
	// } else if(mCurrentFrameOnPhase >= 36.0 && mControlFlag[0] == 1) {
	// 	Eigen::VectorXd diff = skel->getPositionDifferences(skel->getPositions(), mPrevPositions);
	// 	mTarget += diff[1]; //diff.segment<3>(0).norm();
	// }
	

	// punch - force avg 0.55
	// if(mCurrentFrameOnPhase >= 19.0 && mControlFlag[0] == 0) {
	// 	mControlFlag[0] = 1;
	// 	mTarget = 0;
	// 	mTarget2 = 0;
	// } else if(mCurrentFrameOnPhase >= 36.0 && mControlFlag[0] > 0) {
	// 	mTarget /= mTarget2;
	// //  base
	// //	double target_diff = mTarget - 0.6;
	// 	double target_diff = mTarget - 1.0;
	// 	r_target = 2*exp(-pow(target_diff, 2)*5);
	// 	mControlFlag[0] = -1;

	// } else if(mCurrentFrameOnPhase >= 19.0 && mControlFlag[0] > 0) {
	// 	mTarget += mRecordWork.back();
	// 	mTarget2 += 1;

		if(mCurrentFrameOnPhase >= 27 && mControlFlag[0] == 0) {
			Eigen::Vector3d hand = skel->getBodyNode("HandR")->getWorldTransform().translation();
			Eigen::AngleAxisd aa(mHeadRoot.segment<3>(0).norm(), mHeadRoot.segment<3>(0).normalized());
			Eigen::Vector3d target_hand = aa * Eigen::Vector3d(0.65, 0.43, 0.35) + mHeadRoot.segment<3>(3);
			Eigen::Vector3d target_diff = target_hand - hand;
			
			double f_hand = 0;
			for(int i = 0; i < mRecordWork.size(); i++) {
				f_hand += mRecordWork[i];
			}
			f_hand /= mRecordWork.size();
			mRecordWork.clear();
			double f_diff = 1 - f_hand;
			r_target = exp_of_squared(target_diff,0.3) + exp(-pow(f_diff, 2)*0.75);
			mControlFlag[0] = 1;
		}
	// }

	// if(mControlFlag[1] == 0 && mCurrentFrame >= mReferenceManager->GetPhaseLength()) {
	// 	Eigen::VectorXd target_old = mReferenceManager->GetPosition(mCurrentFrame, false);
	// 	Eigen::VectorXd target_diff = skel->getPositionDifferences(this->mTargetPositions, target_old);
	// 	double root_height_diff = this->mTargetPositions[4] - target_old[4];
	// 	Eigen::AngleAxisd root_aa = Eigen::AngleAxisd(mTargetPositions.segment<3>(0).norm(), mTargetPositions.segment<3>(0).normalized());
	// 	Eigen::AngleAxisd root_aa_ = Eigen::AngleAxisd(target_old.segment<3>(0).norm(), target_old.segment<3>(0).normalized());

	// 	Eigen::Vector3d up_vec = root_aa*Eigen::Vector3d::UnitY();
	// 	Eigen::Vector3d up_vec_ = root_aa_*Eigen::Vector3d::UnitY();

	// 	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	// 	double up_vec_angle_ = atan2(std::sqrt(up_vec_[0]*up_vec_[0]+up_vec_[2]*up_vec_[2]),up_vec_[1]);

	// 	double up_vec_angle_diff = up_vec_angle - up_vec_angle_;

	// 	target_diff.head<6>().setZero();
	// 	r_target = 0.5 * (exp_of_squared(target_diff, 0.05) + exp(-pow(root_height_diff, 2)*400) + exp(-pow(up_vec_angle_diff, 2)*100) );
		
	// 	mControlFlag[1] = 1;		
	// }

	return r_target;
}
std::vector<bool> 
Controller::
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

	std::vector<bool> result;
	result.clear();
	for(int i = 0; i < contact.size(); i++) {
		Eigen::Vector3d p = skel->getBodyNode(contact[i])->getWorldTransform().translation();
		if(p[1] < 0.04) {
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
	double r_target = this->GetTargetReward();

	mRewardParts.clear();
	double r_tot = accum_bvh + 10 * r_target;
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(0);
		mRewardParts.push_back(tracking_rewards_bvh[0]);
		mRewardParts.push_back(tracking_rewards_bvh[1]);
		mRewardParts.push_back(tracking_rewards_bvh[2]);
	}
	mTrackingRewardTrajectory += (0.4 * tracking_rewards_bvh[0] + 0.6 * tracking_rewards_bvh[2]);
	if(r_target != 0) mTargetRewardTrajectory = r_target;
}
void
Controller::
UpdateReward()
{
	auto& skel = this->mCharacter->GetSkeleton();
	std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), mTargetPositions,
								 skel->getVelocities(), mTargetVelocities, mRewardBodies, true);
	double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();

	double r_time = exp(-pow(mAdaptiveStep*10,2)*50);

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
UpdateRewardTrajectory() {

	// double r_target = 0;
	// //double r_energy = exp(-pow(mRecordWork.back(), 2));
	// double r_similar = 0; //this->GetSimilarityReward();

	// mTargetRewardTrajectory += (0.5 * r_similar + r_target);
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
	else if(this->nTotalSteps > 500 ) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}
	else if(mOpMode && this->nTotalSteps > mReferenceManager->GetPhaseLength() * 2) {
		mIsTerminal = true;
		terminationReason =  8;
	}

	if(mRecord) {
		if(mIsTerminal) std::cout << terminationReason << std::endl;
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
GetNextPosition(Eigen::VectorXd cur, Eigen::VectorXd delta, Eigen::VectorXd& next) 
{
	Eigen::AngleAxisd cur_root_ori= Eigen::AngleAxisd(cur.segment<3>(0).norm(), cur.segment<3>(0).normalized());
	delta.segment<3>(3) = cur_root_ori * delta.segment<3>(3);
	next.segment<3>(3) = cur.segment<3>(3) + delta.segment<3>(3);

	for(int i = 0; i < mAdaptiveBodies.size(); i++) {
		int idx = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		Eigen::AngleAxisd target_diff_aa = Eigen::AngleAxisd(delta.segment<3>(idx).norm(), delta.segment<3>(idx).normalized());
		Eigen::AngleAxisd cur_aa = Eigen::AngleAxisd(cur.segment<3>(idx).norm(), cur.segment<3>(idx).normalized());
		target_diff_aa = cur_aa * target_diff_aa;
		next.segment<3>(idx) = target_diff_aa.angle() * target_diff_aa.axis();
	}
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
	skel->computeForwardKinematics(true,true,false);
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
	this->mControlFlag.setZero();
	this->nPhase = 0;

	Motion* p_v_target;
	// if(mOpMode) {
	// 	mReferenceManager->GenerateRandomTrajectory(id);
	// 	p_v_target = mReferenceManager->GetMotionForOptimization(mCurrentFrame, id);
	// } else {
		p_v_target = mReferenceManager->GetMotion(mCurrentFrame, isAdaptive);
	// }
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
	this->mRecordTargetPosition.clear();
	this->mRecordBVHPosition.clear();
	this->mRecordRewardPosition.clear();
	this->mRecordObjPosition.clear();
	if(mRecord) {
		Eigen::VectorXd p_obj(mObject->GetSkeleton()->getNumDofs());
		p_obj.setZero();
		p_obj[1] = M_PI;
		p_obj.segment<3>(3) = Eigen::Vector3d(1.1, 0.0, 1.0);
		mObject->GetSkeleton()->setPositions(p_obj);
	}

	SaveStepInfo();

	mHeadRoot = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;
	
	if(isAdaptive)
	{
		Eigen::VectorXd p(mCharacter->GetSkeleton()->getPositions().rows() + 1);
		p << mCharacter->GetSkeleton()->getPositions(), 0;
		data_spline.push_back(std::pair<Eigen::VectorXd,double>(p, mCurrentFrame));
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

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame+1, false);
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->GetPosition(), p_v_target->GetVelocity());
	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	double phase = ((int) mCurrentFrame % mReferenceManager->GetPhaseLength()) / (double) mReferenceManager->GetPhaseLength();
	Eigen::VectorXd state;
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

	for(auto t: mRecordPosition) {
		ofs << t.transpose() << std::endl;
	}
	std::cout << "saved position: " << mRecordPosition.size() << ", "<< mReferenceManager->GetPhaseLength() << ", " << mRecordPosition[0].rows() << std::endl;
	ofs.close();
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
