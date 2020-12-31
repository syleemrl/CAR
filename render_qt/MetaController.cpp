#include "MetaController.h"
namespace DPhy
{	

////////////////////////////////////// SubController //////////////////////////////////////
SubController::SubController()
{}


////////////////////////////////////// MetaController //////////////////////////////////////

MetaController::MetaController()
: mControlHz(30),mSimulationHz(150),mCurrentFrame(0),terminationReason(-1), mIsTerminal(false)
{
	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();

	this->mBaseGravity = Eigen::Vector3d(0,-9.81, 0);
	this->mWorld->setGravity(this->mBaseGravity);

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
}

// 공통
void MetaController::SetAction(action)
{

}
int MetaController::GetNumAction(){return mNumAction;}

Eigen::VectorXd MetaController::GetState()
{
	// 1) 공통 ... 

	// 2) according to mCurrentController...
}


void MetaController::Step()
{
	// 1) 공통 ... 

	// 2) according to mCurrentController...
}


} //end of namespace DPhy


		// Eigen::VectorXd state = this->mController->GetState();

		// p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
		// np::ndarray na = np::from_object(a);
		// Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

		// this->mController->SetAction(action);
		// this->mController->Step();
		// this->mTiming.push_back(this->mController->GetCurrentFrame());