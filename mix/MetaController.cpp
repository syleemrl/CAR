#include "MetaController.h"

namespace DPhy
{	

MetaController::MetaController()
: mControlHz(30),mSimulationHz(150),mCurrentFrame(0), mCurrentFrameOnPhase(0),terminationReason(-1), mIsTerminal(false)
{
	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();

	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81, 0));

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml")).first;
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());

	Eigen::VectorXd kp(this->mCharacter->GetSkeleton()->getNumDofs()), kv(this->mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();
	kv.setZero();
	this->mCharacter->SetPDParameters(kp,kv);

	mInterestedDof = mCharacter->GetSkeleton()->getNumDofs() - 6;
	// mRewardDof = mCharacter->GetSkeleton()->getNumDofs();

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

	// this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();

	mTimeElapsed = 0;
}

// 공통
void MetaController::SetAction(const Eigen::VectorXd& action)
{
	this->mActions = action;
}

Eigen::VectorXd MetaController::GetState()
{
	// 1) 공통 ... 
	// 2) according to mCurrentController...

	bool isAdaptive= true; bool mRecord = true;
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
	double t = GetCurrentRefManager()->GetTimeStep(mCurrentFrameOnPhase, isAdaptive);

	Motion* p_v_target = GetCurrentRefManager()->GetMotion(mCurrentFrame+t, isAdaptive);
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->GetPosition(), p_v_target->GetVelocity()*t);

	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	double phase = ((int) mCurrentFrame % GetCurrentRefManager()->GetPhaseLength()) / (double) GetCurrentRefManager()->GetPhaseLength();
	Eigen::VectorXd state;


	/// 2) according to mCurrentController
	Eigen::VectorXd param = mCurrentController->GetParamGoal();
	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2+param.rows());
	state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase, param;

	return state;
	// if(isParametric) {
	// 	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2+mParamGoal.rows());
	// 	state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase, mParamGoal;
	// }
	// else {
	// 	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2);
	// 	state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase;
	// }

	// return state;
}


void MetaController::Step()
{
	// 1) 공통 ... 
	// 2) according to mCurrentController...
	
	if(IsTerminalState())
		return;

	bool isAdaptive = true;
	bool mRecord= true;

	Eigen::VectorXd s = this->GetState();

	Eigen::VectorXd a = mActions;

	// set action target pos
	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	for(int i = 0; i < mInterestedDof; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.2, -0.7*M_PI, 0.7*M_PI);
	}

	mActions[mInterestedDof] = dart::math::clip(mActions[mInterestedDof]*1.2, -2.0, 1.0);
	mActions[mInterestedDof] = exp(mActions[mInterestedDof]);
	mAdaptiveStep = mActions[mInterestedDof];
	// if(!isAdaptive) mAdaptiveStep = 1;

	mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame += mAdaptiveStep;
	this->mCurrentFrameOnPhase += mAdaptiveStep;
	// nTotalSteps += 1;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	
	Motion* p_v_target = GetCurrentRefManager()->GetMotion(mCurrentFrame, isAdaptive);
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = mCharacter->GetSkeleton()->getPositionDifferences(mTargetPositions, mPrevTargetPositions) / 0.033 * (mCurrentFrame - mPrevFrame);
	delete p_v_target;

	p_v_target = GetCurrentRefManager()->GetMotion(mCurrentFrame, false);
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
	
	for(int i = 0; i < this->mSimPerCon; i += 2){

		for(int j = 0; j < 2; j++) {
			//mCharacter->GetSkeleton()->setSPDTarget(mPDTargetPositions, 600, 49);
			Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
			mCharacter->GetSkeleton()->setForces(torque);
			mWorld->step(false);
		}

		mTimeElapsed += 2 * mAdaptiveStep;
	}


	// if(isAdaptive) {
	// 	this->UpdateAdaptiveReward();
	// }
	// else
	// 	this->UpdateReward();

	this->UpdateTerminalInfo();

	if(mRecord) {
		SaveStepInfo();
	}

	// if(isAdaptive)
	// {
	// 	data_raw.push_back(std::pair<Eigen::VectorXd,double>(mCharacter->GetSkeleton()->getPositions(), mCurrentFrameOnPhase));
	// }

	mPrevTargetPositions = mTargetPositions;
	mPrevFrame = mCurrentFrame;

	if(mPosQueue.size() >= 3)
		mPosQueue.pop();
	if(mTimeQueue.size() >= 3)
		mTimeQueue.pop();
	mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	mTimeQueue.push(mCurrentFrame);

	// if(isAdaptive && mIsTerminal)
	// 	data_raw.clear();

}

void MetaController::UpdateTerminalInfo()
{
	// TODO
	if(mCurrentController->IsTerminalState()) mIsTerminal = true;
}



Eigen::VectorXd 
MetaController::
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

void MetaController::SaveStepInfo()
{
	mRecordBVHPosition.push_back(GetCurrentRefManager()->GetPosition(mCurrentFrame, false));
	mRecordTargetPosition.push_back(mTargetPositions);
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	mRecordCOM.push_back(mCharacter->GetSkeleton()->getCOM());
	mRecordPhase.push_back(mCurrentFrame);

	// if(mRecord) {
	// 	mRecordObjPosition.push_back(mObject->GetSkeleton()->getPositions());
	// }
	// bool rightContact = CheckCollisionWithGround("RightFoot") || CheckCollisionWithGround("RightToe");
	// bool leftContact = CheckCollisionWithGround("LeftFoot") || CheckCollisionWithGround("LeftToe");

	// mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));
}

} //end of namespace DPhy


		// Eigen::VectorXd state = this->mController->GetState();

		// p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
		// np::ndarray na = np::from_object(a);
		// Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

		// this->mController->SetAction(action);
		// this->mController->Step();
		// this->mTiming.push_back(this->mController->GetCurrentFrame());