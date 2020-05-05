#include "Controller.h"
#include "Character.h"
#include <boost/filesystem.hpp>
#include <Eigen/QR>
#include <fstream>
#include <numeric>
#include <algorithm>
namespace DPhy
{	

Controller::Controller(ReferenceManager* ref, std::string stats, bool adaptive, bool record)
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
	
	mAdaptiveBodies.clear();
	mAdaptiveBodies.push_back("Torso");
	mAdaptiveBodies.push_back("Spine");
	mAdaptiveBodies.push_back("FemurR");
	mAdaptiveBodies.push_back("FemurL");
	mAdaptiveBodies.push_back("ArmR");
	mAdaptiveBodies.push_back("ArmL");

	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	this->mCGL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootL"));
	this->mCGR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootR"));
	this->mCGEL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootEndL"));
	this->mCGER = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("FootEndR"));
	this->mCGHL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("HandL"));
	this->mCGHR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("HandR"));
	this->mCGG = collisionEngine->createCollisionGroup(this->mGround.get());

	// pos, time, adaptive angular, adaptive angular root, adaptive linear root
	mActions = Eigen::VectorXd::Zero(this->mInterestedBodies.size()* 3 + 1 + mAdaptiveBodies.size() * 3 + 3);
	
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
	this->mMask.resize(dof);
	this->mMask.setZero();
	
	mControlFlag.resize(2);

	mCount = 0;
	meanTargetReward = 0;

	int num_body_nodes = this->mInterestedBodies.size();
	for(int i = 0; i < num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(mInterestedBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		if( mInterestedBodies[i].find("ForeArmR") != std::string::npos ||
			mInterestedBodies[i].find("ArmR") != std::string::npos ||
			mInterestedBodies[i].find("HandR") != std::string::npos ) {
			// this->mMask[idx] = 1;
			this->mMask.segment<3>(idx) = Eigen::Vector3d::Constant(1);
		}
	}
	
	mRewardLabels.clear();
	if(isAdaptive) {
		mRewardLabels.push_back("total_d");
		mRewardLabels.push_back("total_s");
		mRewardLabels.push_back("tracking");
		mRewardLabels.push_back("linear");
		mRewardLabels.push_back("angular");
		mRewardLabels.push_back("target");
	} else {
		mRewardLabels.push_back("total");
		mRewardLabels.push_back("t_ref");
		mRewardLabels.push_back("t_bvh");
		mRewardLabels.push_back("t_srl");
		mRewardLabels.push_back("action");
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

	double prevAdaptiveStep = mAdaptiveStep;
	mActions[num_body_nodes*3] = dart::math::clip(mActions[num_body_nodes*3]*0.05, -0.8, 0.8);
	mAdaptiveStep = mActions[num_body_nodes*3];
	for(int i = num_body_nodes*3 + 1; i < num_body_nodes*3 + 1 + mAdaptiveBodies.size() * 3; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.02, -0.1, 0.1);
	}
	for(int i = num_body_nodes*3 + 1 + mAdaptiveBodies.size() * 3; i < mActions.size(); i++){
		mActions[i] = dart::math::clip(mActions[i]*0.01, -0.05, 0.05);
	}

	this->mCurrentFrame += (1 + mAdaptiveStep);
	this->mCurrentFrameOnPhase += (1 + mAdaptiveStep);
	if(this->mCurrentFrameOnPhase > mReferenceManager->GetPhaseLength()){
		this->mCurrentFrameOnPhase -= mReferenceManager->GetPhaseLength();
		mHeadRoot = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		mControlFlag.setZero();
	}

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);

	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity() * (1 + mAdaptiveStep);

	delete p_v_target;

	// root linear;
	Eigen::VectorXd prev_target_position = mReferenceManager->GetPosition(mCurrentFrame-(1+mAdaptiveStep));
	Eigen::VectorXd action(mTargetPositions.rows());
	action.setZero();

	int adaptive_idx = num_body_nodes*3 + 1;
	for(int i  = 0; i < mAdaptiveBodies.size(); i++) {
		int idx = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		action.segment<3>(idx) = mActions.segment<3>(adaptive_idx + 3*i);
	}
	action.segment<3>(3) = mActions.tail<3>();
	Eigen::VectorXd delta = mCharacter->GetSkeleton()->getPositionDifferences(mTargetPositions, prev_target_position) + action;	

	GetNextPosition(mPrevTargetPositions, delta, this->mTargetPositions); 

	this->mPDTargetPositions = mTargetPositions;
	this->mPDTargetVelocities = mTargetVelocities;

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
	Eigen::VectorXd torque_sum(this->mCharacter->GetSkeleton()->getNumDofs());
	// Eigen::VectorXd work_sum_joints(this->mCharacter->GetSkeleton()->getNumDofs() / 3 - 2);
	// Eigen::VectorXd torque_sum_joints(this->mCharacter->GetSkeleton()->getNumDofs() / 3 - 2);
	torque_sum.setZero();
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
			torque_sum += torque_masked * 1.0 / mSimulationHz;
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
	nTotalSteps += 1;
	// mRecordWork.push_back(work_sum);
	mRecordTorque.push_back(torque_sum);
	// if(mRecord) {
	// 	this->mRecordTorqueByJoints.push_back(torque_sum_joints);
	// 	mRecordWorkByJoints.push_back(work_sum_joints);
	// }

	if(isAdaptive)
		this->UpdateAdaptiveReward();
	else
		this->UpdateReward();

	this->UpdateTerminalInfo();
	if(mRecord) {
		SaveStepInfo();
	}
	mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;
}
void
Controller::
SaveStepInfo() 
{
	mRecordBVHPosition.push_back(mReferenceManager->GetPosition(mCurrentFrame));
	mRecordTargetPosition.push_back(mTargetPositions);
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	mRecordCOM.push_back(mCharacter->GetSkeleton()->getCOM());
	mRecordTime.push_back(mCurrentFrame);
	bool rightContact = CheckCollisionWithGround("FootEndR") || CheckCollisionWithGround("FootR");
	bool leftContact = CheckCollisionWithGround("FootEndL") || CheckCollisionWithGround("FootL");

	mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));
}
double 
Controller::
ComputeAngularDifferenceFromEllipse(int idx)
{
	Eigen::VectorXd prev_target_position = mReferenceManager->GetPosition(mCurrentFrame-(1+mAdaptiveStep));
	Eigen::VectorXd cur_target_position = mReferenceManager->GetPosition(mCurrentFrame);

	Eigen::Vector3d target_diff_local = JointPositionDifferences(cur_target_position.segment<3>(idx), prev_target_position.segment<3>(idx));
	Eigen::AngleAxisd target_diff_aa = Eigen::AngleAxisd(target_diff_local.norm(), target_diff_local.normalized());
	Eigen::AngleAxisd prev_aa = Eigen::AngleAxisd(mPrevTargetPositions.segment<3>(idx).norm(), mPrevTargetPositions.segment<3>(idx).normalized());
	prev_aa = prev_aa * target_diff_aa;
	Eigen::Vector3d target = prev_aa.angle() * prev_aa.axis();
	
	Eigen::Vector3d diff3d = JointPositionDifferences(target, mTargetPositions.segment<3>(idx));
	Eigen::Vector3d nearest = NearestOnGeodesicCurve3d(target_diff_local.normalized(), Eigen::Vector3d(0, 0, 0), diff3d);

	double x;
	if (target_diff_local.norm() == 0) 
		x = 0;
	else 
		x = nearest.norm();
	double y = (JointPositionDifferences(diff3d, nearest)).norm(); 


	double axis_diff;
	std::vector<std::pair<double, double>> values;
	int total = 0;
	target_diff_local /= (1+mAdaptiveStep);
	for(int i = -3; i <= 3; i++) {
		if (mCurrentFrame+(i-1) < 0 || mCurrentFrame+i > 500) continue;
		else if(i == 0) {
			values.push_back(std::pair<double, double>(0, 1));
			total += target_diff_local.norm();
		} else {
			Eigen::VectorXd prev_target_position2 = mReferenceManager->GetPosition(mCurrentFrame+i-1);
			Eigen::VectorXd cur_target_position2 = mReferenceManager->GetPosition(mCurrentFrame+i);

			Eigen::Vector3d target_diff_local2 = JointPositionDifferences(cur_target_position2.segment<3>(idx), prev_target_position2.segment<3>(idx));
			Eigen::Vector3d nearest2 = NearestOnGeodesicCurve3d(target_diff_local.normalized(), Eigen::Vector3d(0, 0, 0), target_diff_local2);
			double y2 = JointPositionDifferences(target_diff_local2, nearest2).norm();
			// values.push_back(std::pair<double, double>(angle, target_diff_local2.norm() * (1 - abs(i) * 0.3)));
			// total += target_diff_local2.norm() * (1 - abs(i) * 0.3);
			
			values.push_back(std::pair<double, double>(y2, (1 - abs(i) * 0.3)));
			total += (1 - abs(i) * 0.3);
		}
	}
	double dev = 0;
	int total_n = 0;	
	for(int i = 0; i < values.size(); i++) {
		int n = (int)(values[i].second / total * 100.0);
		total_n += n;
		dev += n * values[i].first * values[i].first;
	}
	dev = std::sqrt(dev / (total_n-1.0));
	double size = std::max(target_diff_local.norm(), 0.0075) * 5;
	double max = 2.5 * size;
	double min = 0.5 * size;
	double a = std::max(max - dev * (max-min), min) + 1e-8;
	double b = min + 1e-8;
	double diff = (x*x) / (a*a) + (y*y) / (b*b);

	return diff;
}
double 
Controller::
ComputeLinearDifferenceFromEllipse()
{
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd prev_target_position = mReferenceManager->GetPosition(mCurrentFrame-(1+mAdaptiveStep));
	Eigen::VectorXd cur_target_position = mReferenceManager->GetPosition(mCurrentFrame);

	Eigen::Vector3d target_diff_local = cur_target_position.segment<3>(3) - prev_target_position.segment<3>(3);
	Eigen::AngleAxisd prev_root_ori_target = Eigen::AngleAxisd(prev_target_position.segment<3>(0).norm(), prev_target_position.segment<3>(0).normalized());
	target_diff_local = prev_root_ori_target.inverse() * target_diff_local;

	Eigen::AngleAxisd prev_root_ori = Eigen::AngleAxisd(mPrevTargetPositions.segment<3>(0).norm(), mPrevTargetPositions.segment<3>(0).normalized());
	Eigen::Vector3d target_diff = prev_root_ori * target_diff_local;

	Eigen::Vector3d target = mPrevTargetPositions.segment<3>(3) + target_diff;

	Eigen::Vector3d diff3d = target - mTargetPositions.segment<3>(3);

	double x;
	if (target_diff_local.norm() == 0) 
		x = 0;
	else 
		x = diff3d.dot(target_diff_local.normalized());
	double y = (diff3d - x * target_diff_local.normalized()).norm(); 

	double axis_diff;
	std::vector<std::pair<double, double>> values;
	double total = 0;
	target_diff_local /= (1+mAdaptiveStep);
	for(int i = -3; i <= 3; i++) {
		if (mCurrentFrame+(i-1) < 0 || mCurrentFrame+i > 500) continue;
		else if(i == 0) {
			values.push_back(std::pair<double, double>(0, 1));
			total += target_diff_local.norm();
		} else {
			Eigen::VectorXd prev_target_position2 = mReferenceManager->GetPosition(mCurrentFrame+i-1);
			Eigen::VectorXd cur_target_position2 = mReferenceManager->GetPosition(mCurrentFrame+i);

			Eigen::Vector3d target_diff_local2 = cur_target_position2.segment<3>(3) - prev_target_position2.segment<3>(3);
			Eigen::AngleAxisd prev_root_ori_target2 = Eigen::AngleAxisd(prev_target_position2.segment<3>(0).norm(), prev_target_position2.segment<3>(0).normalized());
			target_diff_local2 = prev_root_ori_target2.inverse() * target_diff_local2;

			double x2 = target_diff_local2.dot(target_diff_local.normalized());
			double y2 = (target_diff_local2 - x2 * target_diff_local.normalized()).norm();

			// if(angle > M_PI / 2.0) angle = M_PI - angle;
			values.push_back(std::pair<double, double>(y2, (1 - abs(i) * 0.3)));
			total += (1 - abs(i) * 0.3);
		}
	}
	double dev = 0;
	int total_n = 0;	
	for(int i = 0; i < values.size(); i++) {
		int n = (int)(values[i].second / total * 100.0);
		total_n += n;
		dev += n * values[i].first * values[i].first;
	}
	dev = std::sqrt(dev / (total_n-1.0));
	double size = std::max(target_diff_local.norm(), 0.0075) * 5;
	double max = 2.5 * size;
	double min = 0.5 * size;
	double a = std::max(max - dev * (max-min), min) + 1e-8;
	double b = min + 1e-8;
	double diff = (x * x) / (a*a) + (y * y) / (b*b);

	return diff;
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
	
	p_diff_reward.resize(list.size()*3);
	for(int i = 0; i < list.size(); i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(list[i])->getParentJoint()->getIndexInSkeleton(0);
		p_diff_reward.segment<3>(3*i) = p_diff.segment<3>(idx);
	}

	Eigen::VectorXd v_diff, v_diff_reward;

	if(useVelocity) {
		v_diff = skel->getVelocityDifferences(velocity, velocity2);
		v_diff_reward.resize(list.size()*3);
		for(int i = 0; i < list.size(); i++){
			int idx = mCharacter->GetSkeleton()->getBodyNode(list[i])->getParentJoint()->getIndexInSkeleton(0);
			v_diff_reward.segment<3>(3*i) = p_diff.segment<3>(idx);
		}
	}

	skel->setPositions(position);
	if(useVelocity) skel->setVelocities(velocity);
	skel->computeForwardKinematics(true,useVelocity,false);

	std::vector<Eigen::Isometry3d> ee_transforms;
	Eigen::VectorXd ee_diff(mEndEffectors.size()*3);
	ee_diff.setZero();	
	for(int i=0;i<mEndEffectors.size();i++){
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

	double sig_p = 0.3 * scale; 
	double sig_v = 1.0 * scale;	
	double sig_com = 0.1 * scale;		
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

	//jump	
	// if(mCurrentFrameOnPhase >= 44.5 && mControlFlag[0] == 0) {
	// 	double target_diff = skel->getCOM()[1] - 1.2;
	// 	r_target = 1.5 * exp(-pow(target_diff, 2) * 20);
	// 	mControlFlag[0] = 1;
	// }
	
	//jump turn	
	if(mCurrentFrameOnPhase >= 36.0 && mControlFlag[0] == 0) {
		mTarget = 0;
		mControlFlag[0] = 1;
	} else if(mCurrentFrameOnPhase >= 68.0 && mControlFlag[0] == 1) {
		mControlFlag[0] = -1;
		double target_diff = mTarget - 5;
		r_target = 2*exp(-pow(target_diff, 2)*0.3);
		meanTargetReward = meanTargetReward * (mCount / (mCount + 1.0)) + r_target * (1.0 / (mCount + 1.0));
		mCount += 1;
	} else if(mCurrentFrameOnPhase >= 36.0 && mControlFlag[0] == 1) {
		Eigen::VectorXd diff = skel->getPositionDifferences(skel->getPositions(), mPrevPositions);
		mTarget += diff[1]; //diff.segment<3>(0).norm();
	}
	
	// punch - position
	// if(mCurrentFrameOnPhase >= 27.0 && mControlFlag[0] == 0) {
	// 	Eigen::Vector3d hand = skel->getBodyNode("HandR")->getWorldTransform().translation();
	// 	hand = hand - mHeadRoot.segment<3>(3);
	// 	Eigen::AngleAxisd root_aa = Eigen::AngleAxisd(mHeadRoot.segment<3>(0).norm(), mHeadRoot.segment<3>(3).normalized());
	// 	hand = root_aa.inverse() * hand;
		
	// 	Eigen::Vector3d target_hand = Eigen::Vector3d(-0.5, 0, 0.4);
	// 	Eigen::Vector3d target_diff = target_hand - hand;
	// 	target_diff[1] = 0;

	// 	r_target = 1.5*exp_of_squared(target_diff,0.3);
	// 	mControlFlag[0] = 1;
	// }

	// punch - force avg 0.55
	// if(mCurrentFrameOnPhase >= 19.0 && mControlFlag[0] == 0) {
	// 	mControlFlag[0] = 1;
	// 	mTarget = 0;
	// 	mTarget2 = 0;
	// } else if(mCurrentFrameOnPhase >= 36.0 && mControlFlag[0] == 1) {
	// 	mTarget /= mTarget2;
	// 	double target_diff = mTarget - 0.8;
	// 	r_target = 1.5*exp(-pow(target_diff, 2)*10);
	// 	mControlFlag[0] = -1;

	// } else if(mCurrentFrameOnPhase >= 19.0 && mControlFlag[0] == 1) {
	// 	mTarget += mRecordTorque.back().norm();
	// 	mTarget2 += 1;
	// }


	if(mControlFlag[1] == 0 && mCurrentFrame >= mReferenceManager->GetPhaseLength()) {
		Eigen::VectorXd target_old = mReferenceManager->GetPosition(mCurrentFrame);
		Eigen::VectorXd target_diff = skel->getPositionDifferences(this->mTargetPositions, target_old);
		double root_height_diff = this->mTargetPositions[4] - target_old[4];
		Eigen::AngleAxisd root_aa = Eigen::AngleAxisd(mTargetPositions.segment<3>(0).norm(), mTargetPositions.segment<3>(0).normalized());
		Eigen::AngleAxisd root_aa_ = Eigen::AngleAxisd(target_old.segment<3>(0).norm(), target_old.segment<3>(0).normalized());

		Eigen::Vector3d up_vec = root_aa*Eigen::Vector3d::UnitY();
		Eigen::Vector3d up_vec_ = root_aa_*Eigen::Vector3d::UnitY();

		double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
		double up_vec_angle_ = atan2(std::sqrt(up_vec_[0]*up_vec_[0]+up_vec_[2]*up_vec_[2]),up_vec_[1]);

		double up_vec_angle_diff = up_vec_angle - up_vec_angle_;

		target_diff.head<6>().setZero();
		r_target = 0.5 * (exp_of_squared(target_diff, 0.05) + exp(-pow(root_height_diff, 2)*400) + exp(-pow(up_vec_angle_diff, 2)*100) );
		
		mControlFlag[1] = 1;		
	}

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
	Eigen::VectorXd dummy = skel->getVelocities();

	std::vector<double> tracking_rewards_ref = this->GetTrackingReward(skel->getPositions(), this->mTargetPositions,
								 dummy, dummy, mRewardBodies, false);
	double accum_ref = std::accumulate(tracking_rewards_ref.begin(), tracking_rewards_ref.end(), 0.0) / tracking_rewards_ref.size();

	double root_linear_diff = ComputeLinearDifferenceFromEllipse();
	double root_angular_diff;
	Eigen::VectorXd joint_angular_diff(this->mAdaptiveBodies.size());
	for(int i = 0; i < this->mAdaptiveBodies.size(); i++) {
		int idx = mCharacter->GetSkeleton()->getBodyNode(mAdaptiveBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		joint_angular_diff[i] = ComputeAngularDifferenceFromEllipse(idx);
	//std::cout << joint_angular_diff[i] << " " << ComputeAngularDifferenceFromEllipse2(idx) << std::endl;
	}


	double r_l = exp(-root_linear_diff*25);
	double r_a = 0;
	if(joint_angular_diff.size() == 0) r_a = 0;
	else {
		for(int i = 0; i < joint_angular_diff.size(); i++) {
			r_a += 1.0 / joint_angular_diff.size() * exp(-joint_angular_diff[i]*25);
		}
	}
	double r_target = this->GetTargetReward();

	std::vector<bool> con_ref = this->GetContactInfo(mTargetPositions);
	std::vector<bool> con_bvh = this->GetContactInfo(mReferenceManager->GetPosition(mCurrentFrame));

	double r_con = 0;
	for(int i = 0; i < con_ref.size(); i++) {
		if(con_ref[i] == con_bvh[i]) {
			r_con += 1;
		}
	}
	r_con /= con_ref.size();

	double r_tot_dense = 0.3 * accum_ref + 0.4 * r_a + 0.2 * r_l + 0.1 * r_con;
 	mRewardParts.clear();
	if(dart::math::isNan(r_tot_dense)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot_dense);
		mRewardParts.push_back(r_target);
		mRewardParts.push_back(accum_ref);
		mRewardParts.push_back(r_l);
		mRewardParts.push_back(r_a);
		mRewardParts.push_back(r_target);
	}
}
void
Controller::
UpdateReward()
{
	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd dummy = skel->getVelocities();
	
	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	Eigen::VectorXd pos_bvh = p_v_target->GetPosition();
	Eigen::VectorXd vel_bvh = p_v_target->GetVelocity();
	delete p_v_target;

	std::vector<double> tracking_rewards_ref = this->GetTrackingReward(skel->getPositions(), this->mTargetPositions,
								 dummy, dummy, mRewardBodies, false);
	double accum_ref = std::accumulate(tracking_rewards_ref.begin(), tracking_rewards_ref.end(), 0.0);

	std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), pos_bvh,
								 dummy, vel_bvh, mRewardBodies, true);
	double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0);

	std::vector<double> tracking_rewards_srl = this->GetTrackingReward(this->mTargetPositions, pos_bvh,
								 dummy, dummy, mAdaptiveBodies, false);

	double accum_srl = std::accumulate(tracking_rewards_srl.begin(), tracking_rewards_srl.end(), 0.0);
	double r_time = exp(-pow(mCurrentFrame - nTotalSteps,2)*5);
	Eigen::VectorXd a = mActions.tail(mAdaptiveBodies.size() * 3 + 3);
	double r_action = exp_of_squared(a, 0.05);
	mRewardParts.clear();

	double r_tot = 0.05 * accum_ref + 0.2 * accum_bvh + 0.15 * accum_srl + 0.1 * r_time + 0.05 * r_action;

	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(accum_ref / tracking_rewards_ref.size());
		mRewardParts.push_back(accum_bvh / tracking_rewards_bvh.size());
		mRewardParts.push_back(accum_srl / tracking_rewards_srl.size());
		mRewardParts.push_back(r_action);
	}

}
void
Controller::
UpdateTerminalInfo()
{	
	Eigen::VectorXd p_ideal;
	if(isAdaptive) 
		p_ideal = mTargetPositions;
	else 
		p_ideal = mReferenceManager->GetPosition(mCurrentFrame);
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];

	Eigen::VectorXd p_old = mReferenceManager->GetPosition(mCurrentFrame);

	Eigen::AngleAxisd root_old_aa = Eigen::AngleAxisd(p_old.segment<3>(0).norm(), p_old.segment<3>(0).normalized());
	Eigen::AngleAxisd root_new_aa = Eigen::AngleAxisd(p_ideal.segment<3>(0).norm(), p_ideal.segment<3>(0).normalized());

	Eigen::Vector3d up_vec1 = root_old_aa*Eigen::Vector3d::UnitY();
	Eigen::Vector3d up_vec2 = root_new_aa*Eigen::Vector3d::UnitY();

	Eigen::VectorXd pos_diff = skel->getPositionDifferences(p_ideal, p_old);
	pos_diff.segment<6>(0).setZero();

	double up_vec_angle_diff = atan2(std::sqrt(up_vec1[0]*up_vec1[0]+up_vec1[2]*up_vec1[2]),up_vec1[1])
							 - atan2(std::sqrt(up_vec2[0]*up_vec2[0]+up_vec2[2]*up_vec2[2]),up_vec2[1]);
	double root_y_diff = p_old[4] - p_ideal[4];

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(p_ideal);
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
	
	if(isAdaptive && !mRecord) {
		if(mCurrentFrameOnPhase < 3.0) {
			if(std::abs(root_y_diff) > 0.3 || std::abs(up_vec_angle_diff) > 0.3 || pos_diff.norm() > 1.5) {
				mIsTerminal = true;
				terminationReason = 6;
			}
		} else if(mRewardParts[5] != 0) {
			if(mRewardParts[5] < meanTargetReward && mRewardParts[5] < 1.8) {
				mIsTerminal = true;
				terminationReason = 7;
			}
		}
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
	this->mCurrentFrameOnPhase = this->mCurrentFrame;
	this->mStartFrame = this->mCurrentFrame;
	this->nTotalSteps = 0;
	this->mTimeElapsed = 0;
	this->mControlFlag.setZero();

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
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

	SaveStepInfo();

	mHeadRoot = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;

	Eigen::AngleAxisd root_aa = Eigen::AngleAxisd(mTargetPositions.segment<3>(0).norm(), mTargetPositions.segment<3>(0).normalized());
	Eigen::Vector3d up_vec = root_aa*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);

	mStartPosition = std::make_tuple(mTargetPositions, mTargetPositions[4], up_vec_angle);

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
void
Controller::
GetNextPosition(Eigen::VectorXd cur, Eigen::VectorXd delta, Eigen::VectorXd& next ) 
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
	
	Eigen::VectorXd cur_target = mReferenceManager->GetPosition(mCurrentFrame);
	Eigen::VectorXd next_target = p_v_target->GetPosition();
	Eigen::VectorXd delta = skel->getPositionDifferences(next_target, cur_target); 

	Eigen::VectorXd next_new = next_target;
	if(mCurrentFrame != 0) {
		this->GetNextPosition(mTargetPositions, delta, next_new);
	} 

	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(next_new, p_v_target->GetVelocity());
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
