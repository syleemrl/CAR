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
	
	this->mCurrentFrameOnPhase = 0;

	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();

	this->mBaseGravity = Eigen::Vector3d(0,-9.81, 0);
	this->mWorld->setGravity(this->mBaseGravity);

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::FCLCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	

	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml")).first;
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);
	
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());

#ifdef OBJECT_TYPE
	std::string object_path = std::string(CAR_DIR)+std::string("/character/") + std::string(OBJECT_TYPE) + std::string(".xml");
	this->mObject = new DPhy::Character(object_path);	
	this->mWorld->addSkeleton(this->mObject->GetSkeleton());
	
	if(isAdaptive){
		auto bn = this->mObject->GetSkeleton()->getBodyNode("Box");
		auto shape_old = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get();
		auto inertia = bn->getInertia();
		inertia.setMass(mParamGoal[0]);
		inertia.setMoment(shape_old->computeInertia(inertia.getMass()));
		bn->setInertia(inertia);
	}

	this->mObject->GetSkeleton()->getBodyNode("Box")->setFrictionCoeff(0.7);
	
	Eigen::VectorXd obj_pos(mObject->GetSkeleton()->getNumDofs());
	obj_pos.setZero();

	mObject->GetSkeleton()->setPositions(obj_pos);
	mObject->GetSkeleton()->setVelocities(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
	mObject->GetSkeleton()->setAccelerations(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
	mObject->GetSkeleton()->computeForwardKinematics(true,false,false);
#endif

	// this->mCharacter->GetSkeleton()->getBodyNode("LeftHand")->setRestitutionCoeff(0.1);	
	// this->mCharacter->GetSkeleton()->getBodyNode("RightHand")->setRestitutionCoeff(0.1);	

	// std::cout<<"HAND default : "<<this->mCharacter->GetSkeleton()->getBodyNode("LeftHand")->getRestitutionCoeff()<<std::endl;
	// std::cout<<"Box default : "<<this->mObject->GetSkeleton()->getBodyNode("Box")->getRestitutionCoeff()<<std::endl;
	// exit(0);

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

#ifdef OBJECT_TYPE
	this->mCGOBJ = collisionEngine->createCollisionGroup(this->mObject->GetSkeleton()->getBodyNode("Box"));
#endif
	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 
	
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
		mRewardLabels.push_back("tracking");
		mRewardLabels.push_back("time");
		mRewardLabels.push_back("similarity");
	} else {
		mRewardLabels.push_back("total");
		mRewardLabels.push_back("p");
		mRewardLabels.push_back("com");
		mRewardLabels.push_back("ee");
		mRewardLabels.push_back("v");
		mRewardLabels.push_back("time");
	}

	// } else if(isAdaptive && mRecord) {
	// 	path = std::string(CAR_DIR)+std::string("/character/sandbag.xml");
	// 	this->mObject = new DPhy::Character(path);	
	// 	this->mWorld->addSkeleton(this->mObject->GetSkeleton());
	// }
	foot_diff = std::vector<double>();
	pr_calculated= false;

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

	if(mCurrentFrameOnPhase >=21 && !left_detached && !leftHandConstraint) attachHandToBar(true, Eigen::Vector3d(0.06, -0.025, 0));
	else if(mCurrentFrameOnPhase >=190 && leftHandConstraint) { removeHandFromBar(true); left_detached= true; }

	if(mCurrentFrameOnPhase >=21 && !right_detached && !rightHandConstraint) attachHandToBar(false, Eigen::Vector3d(-0.06, -0.025, 0));
	else if(mCurrentFrameOnPhase >=190 && rightHandConstraint) {removeHandFromBar(false); right_detached =true;}


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

	mActions[mInterestedDof] = dart::math::clip(mActions[mInterestedDof]*1.2, -2.0, 1.0);
	mActions[mInterestedDof] = exp(mActions[mInterestedDof]);
	mAdaptiveStep = mActions[mInterestedDof];
	if(!isAdaptive) mAdaptiveStep = 1;

	mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame += (mAdaptiveStep);
	this->mCurrentFrameOnPhase += (mAdaptiveStep);
	nTotalSteps += 1;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	
	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame, isAdaptive);
	Eigen::VectorXd p_now = p_v_target->GetPosition();
	// p_now[4] -= (mDefaultRootZero[4]- mRootZero[4]);
	// p_now.segment<3>(3) = p_now.segment<3>(3)- (mDefaultRootZero.segment<3>(3)- mRootZero.segment<3>(3));
	this->mTargetPositions = p_now ; //p_v_target->GetPosition();
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
	
	impulse_on_wrong_body = 0;
	mSumTorque.resize(dof);
	mSumTorque.setZero();
	for(int i = 0; i < this->mSimPerCon; i += 2){

		for(int j = 0; j < 2; j++) {
			//mCharacter->GetSkeleton()->setSPDTarget(mPDTargetPositions, 600, 49);
			
			// torque limit
			Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
			// for(int j = 0; j < num_body_nodes; j++) {
			// 	int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
			// 	int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			// 	std::string name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
			// 	double torquelim = mCharacter->GetTorqueLimit(name) * 1.5;
			// 	double torque_norm = torque.block(idx, 0, dof, 1).norm();
			
			// 	torque.block(idx, 0, dof, 1) = std::max(-torquelim, std::min(torquelim, torque_norm)) * torque.block(idx, 0, dof, 1).normalized();
			// }

			//mCharacter->GetSkeleton()->setForces(torque);

			mWorld->step(false);
			mSumTorque += torque.cwiseAbs();

		}

		for(auto bn: this->mCharacter->GetSkeleton()->getBodyNodes()){
			if(bn->getName()== "LeftToe" || bn->getName()=="RightToe"|| bn->getName()== "LeftFoot" || bn->getName()=="RightFoot"|| bn->getName()=="LeftHand" || bn->getName()== "RightHand") continue;
			impulse_on_wrong_body += bn->getConstraintImpulse().norm();
		}
		// head_force+= this->mCharacter->GetSkeleton()->getBodyNode("Head")->getConstraintImpulse();
		// leftToe_force+= this->mCharacter->GetSkeleton()->getBodyNode("LeftToe")->getConstraintImpulse();
		// rightToe_force+= this->mCharacter->GetSkeleton()->getBodyNode("RightToe")->getConstraintImpulse();

		// leftFoot_force+= this->mCharacter->GetSkeleton()->getBodyNode("LeftFoot")->getConstraintImpulse();
		// rightFoot_force+= this->mCharacter->GetSkeleton()->getBodyNode("RightFoot")->getConstraintImpulse();

		mTimeElapsed += 2 * (mAdaptiveStep);
	}

	// std::cout<<"impulse_on_wrong_body : "<<impulse_on_wrong_body<<std::endl;

	// dart::collision::DistanceOption option;
	// option.enableNearestPoints = true;
	// dart::collision::DistanceResult result;

// #ifdef OBJECT_TYPE
// 	result.clear();
// 	mCGOBJ->distance(mCGHR.get(), option, &result);
// 	this->hr_dist= result.minDistance;
	
// 	result.clear();
// 	mCGOBJ->distance(mCGHL.get(), option, &result);
// 	this->hl_dist= result.minDistance;
// #endif

	// std::cout<<mCurrentFrame<<"\tmin: "<<hl_dist<<" "<<hr_dist<<std::endl;

	if(this->mCurrentFrameOnPhase > mReferenceManager->GetPhaseLength()){
		this->mCurrentFrameOnPhase -= mReferenceManager->GetPhaseLength();
		mRootZero = mCharacter->GetSkeleton()->getPositions().segment<6>(0);		
		// stickLeftFoot = mCharacter->GetSkeleton()->getBodyNode("LeftFoot")->getWorldTransform().translation();
		// stickRightFoot = mCharacter->GetSkeleton()->getBodyNode("RightFoot")->getWorldTransform().translation();

		mDefaultRootZero = mReferenceManager->GetMotion(mCurrentFrame, true)->GetPosition().segment<6>(0);
		mRootZeroDiff = mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase, false)->GetPosition().segment<3>(3);

		this->mStartRoot = this->mCharacter->GetSkeleton()->getPositions().segment<3>(3);
		Eigen::Vector3d lf = this->mCharacter->GetSkeleton()->getBodyNode("LeftFoot")->getWorldTransform().translation();
		Eigen::Vector3d rf = this->mCharacter->GetSkeleton()->getBodyNode("RightFoot")->getWorldTransform().translation();
		Eigen::Vector3d mf = (lf+rf)/2.; 
		this->mStartFoot = Eigen::Vector3d(mf[0], std::min(lf[1], rf[1]), mf[2]);

		if(isAdaptive) {
			mTrackingRewardTrajectory /= mCountTracking;

			mFitness.sum_contact/= mCountTracking;
			mFitness.sum_pos/= mCountTracking;
			mFitness.sum_vel/= mCountTracking;
			mFitness.sum_fall /= mCountFall;
			// mFitness.sum_slide/= mCountTracking;

			// std::cout<<mCurrentFrame<<" : "<<mTrackingRewardTrajectory<<std::endl;
			//if(mCurrentFrame < 2*mReferenceManager->GetPhaseLength() ){
				// std::cout<<"f: "<<mCurrentFrame<<"/fop: "<<mCurrentFrameOnPhase<<" / "<<(mReferenceManager->GetPhaseLength())<<std::endl;
			mReferenceManager->SaveTrajectories(data_raw, std::tuple<double, double, Fitness>(mTrackingRewardTrajectory, mParamRewardTrajectory, mFitness), mParamCur);
			//}
			data_raw.clear();

			mFitness.sum_contact = 0;
			// mFitness.sum_slide = 0;
			auto& skel = mCharacter->GetSkeleton();
			mFitness.sum_pos.resize(skel->getNumDofs());
			mFitness.sum_vel.resize(skel->getNumDofs());
			mFitness.sum_pos.setZero();
			mFitness.sum_vel.setZero();
			mFitness.sum_reward = 0;
			mFitness.sum_fall = 0;
			
			mTrackingRewardTrajectory = 0;
			mParamRewardTrajectory = 0;

			mControlFlag.setZero();
			mCountParam = 0;
			mCountTracking = 0;
			
			// mEnergy.setZero();
			// mVelocity = 0;
			foot_diff.clear();	
			pr_calculated = false;

			mCountFall = 0;
			mRootFall = 0;
			mStickFoot = 0;

		}

		#ifdef OBJECT_TYPE
		// place the object according to the mRootZero
		Eigen::Vector3d root_diff = mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase, false)->GetPosition().segment<3>(3);

		int box_dof =mObject->GetSkeleton()->getNumDofs();
		Eigen::VectorXd obj_pos(box_dof);
		obj_pos.setZero();
		obj_pos[box_dof-1]= root_diff[2]; // z-direction
		obj_pos[box_dof-3] = root_diff[0];

		mObject->GetSkeleton()->setPositions(obj_pos);
		mObject->GetSkeleton()->setVelocities(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->setAccelerations(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
		mObject->GetSkeleton()->computeForwardKinematics(true,false,false);
		this->mObjectStartPosition = mObject->GetSkeleton()->getBodyNode("Box")->getWorldTransform().translation()[2];
		left_detached = false;
		right_detached = false;
		#endif

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
		data_raw.push_back(std::pair<Eigen::VectorXd,double>(mCharacter->GetSkeleton()->getPositions(), mCurrentFrameOnPhase));
	}
	if(mPosQueue.size() >= 3)
		mPosQueue.pop();
	if(mTimeQueue.size() >= 3)
		mTimeQueue.pop();
	mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	mTimeQueue.push(mCurrentFrame);
	mPrevTargetPositions = mTargetPositions;

	if(isAdaptive && mIsTerminal)
		data_raw.clear();

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

	#ifdef OBJECT_TYPE
	mRecordObjPosition.push_back(mObject->GetSkeleton()->getPositions());
	#endif
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

	this->mControlFlag.resize(4);
	this->mControlFlag.setZero();

	mCountParam = 0;
	mCountTracking = 0;
	data_raw.clear();
	mFitness.sum_reward = 0;

	mCountFall = 0;
	mRootFall = 0;
	mStickFoot = 0;
	mFitness.sum_fall = 0;
	// mEnergy.setZero();
	// mVelocity = 0;
	// mPrevVelocity = 0;
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

#ifdef OBJECT_TYPE
	Eigen::Vector3d lh_end = mCharacter->GetSkeleton()->getBodyNode("LeftHand")->getWorldTransform()* Eigen::Vector3d(0.06, -0.025, 0);
	Eigen::Vector3d rh_end = mCharacter->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform()* Eigen::Vector3d(-0.06, -0.025, 0);
	double box_end_z = mObject->GetSkeleton()->getBodyNode("Box")->getWorldTransform().translation()[2] - 0.5;
	// std::cout<<mCurrentFrame<<"/ hand: "<<lh_end[2]<<", "<<rh_end[2]<<"/ box: "<<box_end_z<<std::endl;
	Eigen::VectorXd hand_diff(2); hand_diff.setZero();
	if(mCurrentFrameOnPhase >=28 && mCurrentFrameOnPhase<=190) hand_diff << lh_end[2]- box_end_z, rh_end[2]-box_end_z;
#endif

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
	double sig_ee = 0.5 * scale;		

	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_v;
	if(useVelocity)
	{
		r_v = exp_of_squared(v_diff_reward,sig_v);
	}
	double r_ee = exp_of_squared(ee_diff,sig_ee);
	double r_com = exp_of_squared(com_diff,sig_com);

	// BOX PUSHING ONLY
	// Eigen::VectorXd hand_diff(2); hand_diff.setZero();
	// if(mCurrentFrameOnPhase >=28 && mCurrentFrameOnPhase<=190) hand_diff << hl_dist, hr_dist;
	// double r_hand = exp_of_squared(hand_diff, sig_hand);
	// r_ee = (r_ee+r_hand)/2.;

	// std::cout<<mCurrentFrameOnPhase<<"\t"<<hand_diff.transpose()<<"\t"<<r_hand<<std::endl;

#ifdef OBJECT_TYPE
	double sig_hand = 0.07* scale;		
	double r_hand = exp_of_squared(hand_diff, sig_hand);
	// std::cout<<mCurrentFrame<<" "<<hand_diff.transpose()<<", r_hand: "<<r_hand<<std::endl;
	r_ee = 0.7*r_ee + 0.3*r_hand; 
#endif

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
GetContactInfo(Eigen::VectorXd pos, double base_height) 
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
			p[1]-= base_height;
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
	vel *= (mCurrentFrame - mPrevFrame); 
	delete p_v_target;

	double ref_obj_height = 0; 
	double cur_obj_height = 0; 

	std::vector<std::pair<bool, Eigen::Vector3d>> contacts_ref = GetContactInfo(pos, ref_obj_height);
	std::vector<std::pair<bool, Eigen::Vector3d>> contacts_cur = GetContactInfo(skel->getPositions(), cur_obj_height);
	double con_diff = 0;

	for(int i = 0; i < contacts_cur.size(); i++) {
		if(contacts_ref[i].first || contacts_cur[i].first) {
			con_diff += pow(((contacts_cur[i].second)(1) - (contacts_ref[i].second)(1))*3, 2);
		}
	}

	// sliding prevention //TODO
	// double footSlide = 0;
	// double r_footSlide = 1;
	// if(mCurrentFrameOnPhase <=29 || (mCurrentFrameOnPhase >=50 && stickFoot)){
	// 	Eigen::Vector3d lf = skel->getBodyNode("LeftFoot")->getWorldTransform().translation();
	// 	Eigen::Vector3d rf = skel->getBodyNode("RightFoot")->getWorldTransform().translation();

	// 	Eigen::VectorXd foot_diff (6);
	// 	foot_diff << (lf-stickLeftFoot), (rf- stickRightFoot);

	// 	footSlide = foot_diff.norm();
	// 	r_footSlide = exp_of_squared(foot_diff, 0.1);
	// 	// std::cout<<mCurrentFrame<<" : "<<r_footSlide<<" ( "<<foot_diff.transpose()<<") \n";
	// }else if (mCurrentFrameOnPhase >29 && stickFoot) stickFoot = false;
	// else if(mCurrentFrameOnPhase>= 50 && !stickFoot){

	// 	stickLeftFoot = skel->getBodyNode("LeftFoot")->getWorldTransform().translation();
	// 	stickRightFoot = skel->getBodyNode("RightFoot")->getWorldTransform().translation();
	// 	stickFoot = true;
	// }

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
		if(name.compare("LeftHand")==0 ||name.compare("RightHand")==0){
			p_diff.segment<3>(idx) *= 5;
			// v_diff.segment<3>(idx) *= 5;
			// v_diff.segment<3>(idx + 3) *= 10;
		}
		if(name.compare("Hips") == 0 ) {
			p_diff.segment<3>(idx) *= 2;
			p_diff.segment<3>(idx + 3) *= 5;
			if(mCurrentFrameOnPhase >= 215) {
				p_diff.segment<3>(idx) *= 5;
				p_diff.segment<3>(idx + 3) *= 5;
				mFitness.sum_fall += p_aligned(0);
				mCountFall += 1;
			}
			// v_diff.segment<3>(idx) *= 5;
			// v_diff.segment<3>(idx + 3) *= 10;
		} 
	}

	double r_con = exp(-abs(con_diff));
	double r_ee = exp_of_squared(v_diff, 3);
	double r_p = exp_of_squared(p_diff,0.3);
	mPrevFrame2 = mPrevFrame;
	mPrevFrame = mCurrentFrame;

	mFitness.sum_pos += p_diff.cwiseAbs(); 
	mFitness.sum_vel += v_diff.cwiseAbs();
	mFitness.sum_contact += abs(con_diff);
	// mFitness.sum_slide += footSlide;
	
	return r_con  * r_p * r_ee ; //* r_footSlide;
}
double 
Controller::
GetParamReward()
{
	double r_param = 0;

	if(mCurrentFrameOnPhase >= 200 && !pr_calculated){
		double cur_obj_z= mObject->GetSkeleton()->getBodyNode("Box")->getWorldTransform().translation()[2];
		double travel_distance = cur_obj_z - mObjectStartPosition;
		double distance_diff= travel_distance - (4-0.7);
		r_param = exp_of_squared(distance_diff, 0.65);

		if(abs(distance_diff) < 0.5) {
			mParamCur = mParamGoal;
		} else {
			mParamCur(0) = -1;
		}
		pr_calculated = true;

		if(mRecord) std::cout<<travel_distance<<"/ goal: " <<mParamGoal<<"/ cur: "<<mParamCur.transpose()<<" / r: "<<r_param<<std::endl;
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
	double time_diff = (mAdaptiveStep) - mReferenceManager->GetTimeStep(mPrevFrameOnPhase, true);
	double r_time = exp(-pow(time_diff, 2)*75);

	double r_tracking = 0.85 * accum_bvh + 0.15 * r_time;
	double r_similarity = this->GetSimilarityReward();
	double r_param = this->GetParamReward();
	
	mSumTorque /= mSimPerCon;
	double r_torque = exp_of_squared(mSumTorque, 50);

	double r_tot = 0.99*r_tracking +0.01*r_torque;

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

	double r_time = exp(-pow((mActions[mInterestedDof]-1),2)*40);
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

	Eigen::Vector3d lf = mCharacter->GetSkeleton()->getBodyNode("LeftUpLeg")->getWorldTransform().translation();
	Eigen::Vector3d rf = mCharacter->GetSkeleton()->getBodyNode("RightUpLeg")->getWorldTransform().translation();
	Eigen::Vector3d ls = mCharacter->GetSkeleton()->getBodyNode("LeftShoulder")->getWorldTransform().translation();
	Eigen::Vector3d rs = mCharacter->GetSkeleton()->getBodyNode("RightShoulder")->getWorldTransform().translation();
	Eigen::Vector3d right_vector = ((rf-lf)+(rs-ls))/2.;
	right_vector[1]= 0;
	Eigen::Vector3d forward_vector=  Eigen::Vector3d::UnitY().cross(right_vector);
	double forward_angle= std::atan2(forward_vector[0], forward_vector[2]);

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

	if(!mRecord && root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 2;
	}
	if(!mRecord && (root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT)){
		mIsTerminal = true;
		terminationReason = 1;
	}
	else if(!mRecord && std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 5;
	}
	else if(mCurrentFrame > mReferenceManager->GetPhaseLength()* 2+ 10) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}

	// bool head_collide_box = CheckCollisionWithObject("Head");
	// if(head_collide_box) {
	// 	mIsTerminal = true;
	// 	terminationReason = 9;
	// }
	double box_end_z = mObject->GetSkeleton()->getBodyNode("Box")->getWorldTransform().translation()[2] - 0.5;
	double lf_z = skel->getBodyNode("LeftFoot")->getWorldTransform().translation()[2];
	lf_z = std::max(skel->getBodyNode("LeftToe")->getWorldTransform().translation()[2], lf_z);

	double rf_z = 0.5 * skel->getBodyNode("RightFoot")->getWorldTransform().translation()[2];
	rf_z = std::max(skel->getBodyNode("RightToe")->getWorldTransform().translation()[2], rf_z);

	// if(mCurrentFrameOnPhase >= 40 &&  mCurrentFrameOnPhase <= 200 && (lf_z + 0.15 > box_end_z || rf_z + 0.15 > box_end_z)) {
	// 	mIsTerminal = true;
	// 	terminationReason = 10;
	// }
	if(impulse_on_wrong_body > 10){
		mIsTerminal = true;
		terminationReason = 10;

	}
	if(mRecord) {
		if(mIsTerminal) std::cout << "terminate Reason : "<<terminationReason << std::endl;
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
		mFitness.sum_contact = 0;
		mFitness.sum_slide = 0;
		mFitness.sum_pos.resize(skel->getNumDofs());
		mFitness.sum_vel.resize(skel->getNumDofs());
		mFitness.sum_pos.setZero();
		mFitness.sum_vel.setZero();
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

	// Eigen::VectorXd nextTargetPositions = mReferenceManager->GetPosition(mCurrentFrame+1, isAdaptive);
	// this->mTargetVelocities = mCharacter->GetSkeleton()->getPositionDifferences(nextTargetPositions, mTargetPositions) / 0.033;
	// std::cout <<  mCharacter->GetSkeleton()->getPositionDifferences(nextTargetPositions, mTargetPositions).segment<3>(3).transpose() << std::endl;
	this->mPDTargetPositions = mTargetPositions;
	this->mPDTargetVelocities = mTargetVelocities;

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;
	ClearRecord();
	SaveStepInfo();

	Eigen::VectorXd tl_cur(3 + mEndEffectors.size() * 3);
	tl_cur.segment<3>(0) = skel->getRootBodyNode()->getWorldTransform().translation();
	for(int i = 0; i < mEndEffectors.size(); i++) {
		tl_cur.segment<3>(i*3 + 3) = skel->getBodyNode(mEndEffectors[i])->getWorldTransform().translation();
	}

	mRootZero = mTargetPositions.segment<6>(0);
	this->mRootZeroDiff= mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase, false)->GetPosition().segment<3>(3);

	stickLeftFoot = skel->getBodyNode("LeftFoot")->getWorldTransform().translation();
	stickRightFoot = skel->getBodyNode("RightFoot")->getWorldTransform().translation();
	stickFoot = true;
	// mFootZero = (lf+rf)/2.;
	mDefaultRootZero = mRootZero; 

	mTlPrev2 = mTlPrev;
	mTlPrev = tl_cur;	

	mPrevFrame = mCurrentFrame;
	mPrevFrame2 = mPrevFrame;

	while(!mPosQueue.empty())
		mPosQueue.pop();
	while(!mTimeQueue.empty())
		mTimeQueue.pop();
	mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	mTimeQueue.push(0);
	mAdaptiveStep = 1;

	mPrevTargetPositions = mTargetPositions;
	
	if(isAdaptive)
	{
		data_raw.push_back(std::pair<Eigen::VectorXd,double>(mCharacter->GetSkeleton()->getPositions(), mCurrentFrame));
	
		auto bn = this->mObject->GetSkeleton()->getBodyNode("Box");
		auto shape_old = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get();
		auto inertia = bn->getInertia();
		inertia.setMass(mParamGoal[0]);
		inertia.setMoment(shape_old->computeInertia(inertia.getMass()));
		bn->setInertia(inertia);
	}

	#ifdef OBJECT_TYPE
	// place the object according to current Param Goal
	
	Eigen::VectorXd obj_pos(mObject->GetSkeleton()->getNumDofs());
	obj_pos.setZero();

	this->mObject->GetSkeleton()->setPositions(obj_pos);
	this->mObject->GetSkeleton()->setVelocities(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
	this->mObject->GetSkeleton()->setAccelerations(Eigen::VectorXd::Zero(mObject->GetSkeleton()->getNumDofs()));
	this->mObject->GetSkeleton()->computeForwardKinematics(true,false,false);

	this->mStartRoot = this->mCharacter->GetSkeleton()->getPositions().segment<3>(3);
	this->mRootZeroDiff= mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase, false)->GetPosition().segment<3>(3);

	Eigen::Vector3d lf = this->mCharacter->GetSkeleton()->getBodyNode("LeftFoot")->getWorldTransform().translation();
	Eigen::Vector3d rf = this->mCharacter->GetSkeleton()->getBodyNode("RightFoot")->getWorldTransform().translation();
	Eigen::Vector3d mf = (lf+rf)/2.; 
	this->mStartFoot = Eigen::Vector3d(mf[0], std::min(lf[1], rf[1]), mf[2]);
	this->mObjectStartPosition = mObject->GetSkeleton()->getBodyNode("Box")->getWorldTransform().translation()[2];

	#endif

	foot_diff.clear();
	pr_calculated= false;


	if(leftHandConstraint && mCurrentFrame <=20 ) removeHandFromBar(true);
	if(rightHandConstraint && mCurrentFrame <=20) removeHandFromBar(false);

	//45, 59
	left_detached= (mCurrentFrame >=190) ? true: false; 
	right_detached= (mCurrentFrame >=190) ? true: false;
	// std::cout<<"controller, placed : "<<this->mObject->GetSkeleton()->getPositions().transpose()<<std::endl;

	//0: -8.63835e-05      1.04059     0.016015 / 41 : 0.00327486    1.34454   0.378879 / 81 : -0.0177552    1.48029   0.614314

	// std::cout<<"RSI : "<<mCurrentFrame<<std::endl;
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

	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(pos);
	skel->setVelocities(vel);
	skel->computeForwardKinematics(true, true, false);

	ret.resize((num_ee)*12+15);
//	ret.resize((num_ee)*9+12);

	for(int i=0;i<num_ee;i++)
	{		
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		//Eigen::Quaterniond q(transform.linear());
		// Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
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

bool
Controller::
CheckCollisionWithObject(std::string bodyName){
	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	dart::collision::CollisionOption option;
	dart::collision::CollisionResult result;
	if(bodyName == "RightFoot"){
		bool isCollide = collisionEngine->collide(this->mCGR.get(), this->mCGOBJ.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftFoot"){
		bool isCollide = collisionEngine->collide(this->mCGL.get(), this->mCGOBJ.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "RightToe"){
		bool isCollide = collisionEngine->collide(this->mCGER.get(), this->mCGOBJ.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftToe"){
		bool isCollide = collisionEngine->collide(this->mCGEL.get(), this->mCGOBJ.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "RightHand"){
		bool isCollide = collisionEngine->collide(this->mCGHR.get(), this->mCGOBJ.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftHand"){
		bool isCollide = collisionEngine->collide(this->mCGHL.get(), this->mCGOBJ.get(), option, &result);
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
		//	ret.segment<6>(6*i) << rot, transform.translation();
		p.segment<6>(6*(i-1)) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
								 transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2);
	}

	v = v_save;

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
	Eigen::VectorXd p_now = p_v_target->GetPosition();
	// p_now[4] -= (mDefaultRootZero[4]- mRootZero[4]);
	// p_now.segment<3>(3) = p_now.segment<3>(3)- (mDefaultRootZero.segment<3>(3)- mRootZero.segment<3>(3));
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_now, p_v_target->GetVelocity()*t);
	// Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->GetPosition(), p_v_target->GetVelocity()*t);

	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	double phase = ((int) mCurrentFrame % mReferenceManager->GetPhaseLength()) / (double) mReferenceManager->GetPhaseLength();
	Eigen::VectorXd state;

	double com_diff = 0;
	if(isParametric) {
		state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2+mParamGoal.rows());
		// state<< p, v, up_vec_angle, root_height, p_next, ee, mCurrentFrameOnPhase, mParamGoal;
		state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase, mParamGoal;
	}
	else {
		state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2);
		// state<< p, v, up_vec_angle, root_height, p_next, ee, mCurrentFrameOnPhase;
		state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase;
	}

	// if(mRecord && mCurrentFrame < 10){
	// 	std::cout<<"i :: "<<mCurrentFrame<<std::endl;
	// 	std::cout<< "p : "<< p.segment<6>(0).transpose() <<std::endl;
	// 	std::cout<< "v : "<< v.segment<6>(0).transpose() <<std::endl;
	// 	std::cout<< up_vec_angle <<std::endl;
	// 	std::cout<< root_height <<std::endl;
	// 	std::cout<< "p_next : "<<p_next.segment<6>(0).transpose() <<std::endl;
	// 	std::cout<< "ee : "<<ee.transpose() <<std::endl;
	// 	std::cout<< "mCurrentFrameOnPhase: "<<mCurrentFrameOnPhase <<std::endl;
	// 	std::cout<< "mParamGoal: "<<mParamGoal.transpose() <<std::endl;
	// 	std::cout<<std::endl;
		
	// 	// bool rightContact = CheckCollisionWithGround("RightFoot") || CheckCollisionWithGround("RightToe") || CheckCollisionWithObject("RightFoot") || CheckCollisionWithObject("RightToe");
	// 	// bool leftContact = CheckCollisionWithGround("LeftFoot") || CheckCollisionWithGround("LeftToe") ||  CheckCollisionWithObject("LeftFoot") || CheckCollisionWithObject("LeftToe");

	// 	bool contactWithObj =  CheckCollisionWithObject("LeftFoot") || CheckCollisionWithObject("LeftToe") || CheckCollisionWithObject("RightFoot") || CheckCollisionWithObject("RightToe") ||CheckCollisionWithObject("LeftHand") || CheckCollisionWithObject("RightHand") ;
	// 	if(contactWithObj) std::cout<<" COLLISION WITH OBJECT ----------------------------------------------- "<<std::endl;
	// }

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

void Controller::attachHandToBar(bool left, Eigen::Vector3d offset){

	std::string hand = (left) ? "LeftHand" : "RightHand";
	dart::dynamics::BodyNodePtr hand_bn = this->mCharacter->GetSkeleton()->getBodyNode(hand);
	dart::dynamics::BodyNodePtr bar_bn = this->mObject->GetSkeleton()->getBodyNode("Box");
	Eigen::Vector3d jointPos = hand_bn->getTransform() * offset;

	double box_end_z = mObject->GetSkeleton()->getBodyNode("Box")->getWorldTransform().translation()[2] - 0.5;
	double distance = std::pow(jointPos[2]- box_end_z, 2.0);

	if(mRecord)	std::cout<<mCurrentFrameOnPhase<<" "<<box_end_z<<" "<<jointPos[2]<<" : "<<distance<<std::endl;
	if(distance >0.05) return;
	// if(distance > 0.05 || jointPos[2] < 3.5 || jointPos[2] > 3.7 || jointPos[1] > 0.95) return;

	// std::cout<<"success"<<std::endl;

	if(left && leftHandConstraint) removeHandFromBar(true);
	else if(!left && rightHandConstraint) removeHandFromBar(false);

	dart::constraint::BallJointConstraintPtr cl = std::make_shared<dart::constraint::BallJointConstraint>( hand_bn, bar_bn, jointPos);
	this->mWorld->getConstraintSolver()->addConstraint(cl);

	if(left) leftHandConstraint = cl;
	else rightHandConstraint = cl;

	if(mRecord)	{
		std::cout<<"attach "<<mCurrentFrameOnPhase<<" ";
		if(left) std::cout<<"left : ";
		else std::cout<<"right : ";
		std::cout<<jointPos.transpose()<<std::endl;		
	}

}

void Controller::removeHandFromBar(bool left){
	// std::cout<<"REMOVE "<<left<<std::endl;
	if(left && leftHandConstraint) {
	    mWorld->getConstraintSolver()->removeConstraint(leftHandConstraint);
	    leftHandConstraint = nullptr;
    	
	}else if(!left && rightHandConstraint){
	    mWorld->getConstraintSolver()->removeConstraint(rightHandConstraint);
    	rightHandConstraint = nullptr;
		
	}

	if(mRecord){
		std::cout<<"remove "<<mCurrentFrameOnPhase<<" ";
		if(left) std::cout<<"left : "<<std::endl;
		else std::cout<<"right : "<<std::endl;		
	}
}

}
