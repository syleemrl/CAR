#include "MetaController.h"
#include <tinyxml.h>

namespace DPhy
{	

MetaController::MetaController()
: mControlHz(30),mSimulationHz(150)
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

	LoadControllers();
	mCurrentController = mSubControllers["Idle"];
	mPrevController = nullptr;
}
void MetaController::LoadControllers()
{
	std::string ctrl_path = std::string(CAR_DIR)+ std::string("/scene/s1_ctrl.xml");
	std::cout<<"loadControllers: "<<ctrl_path<<std::endl;
	TiXmlDocument doc;
	if(!doc.LoadFile(ctrl_path)){
		std::cout << "Can't open scene file : " << ctrl_path << std::endl;
	}

	TiXmlElement *skeldoc = doc.FirstChildElement("ControllerList");
	
	for(TiXmlElement *body = skeldoc->FirstChildElement("SubController"); body != nullptr; body = body->NextSiblingElement("SubController")){
		
		std::string ctrl_type = body->Attribute("type");
		std::string ctrl_bvh = body->Attribute("bvh");
		std::string ctrl_ppo = body->Attribute("ppo");

		std::cout<< "================ ADD SUB Controller: "<<ctrl_type<<" :: "<<ctrl_bvh<<" , "<<ctrl_ppo<<std::endl;
		SubController* newSC;
		 if(ctrl_type == "Punch"){
			newSC = new PUNCH_Controller(ctrl_bvh, ctrl_ppo);
		}else if(ctrl_type == "Kick"){
			newSC = new KICK_Controller(ctrl_bvh, ctrl_ppo);
		}else if(ctrl_type == "Idle"){
			newSC = new IDLE_Controller(ctrl_bvh, ctrl_ppo);	
		}else if(ctrl_type == "Block"){
			newSC = new BLOCK_Controller(ctrl_bvh, ctrl_ppo);	
		} else if(ctrl_type == "Pivot"){
			newSC = new PIVOT_Controller(ctrl_bvh, ctrl_ppo);	
		}else{
			std::cout<<" NOT A PROPER COTNROLLER TYPE : "<<ctrl_type<<std::endl;
			continue;
		}

		newSC->mParamGoal = newSC->mReferenceManager->GetParamGoal();
		AddSubController(newSC);
	}
}
void MetaController::Reset()
{
	this->mWorld->reset();
	auto& skel = mCharacter->GetSkeleton();
	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();

	this->mTotalSteps = 0; 
	mCurrentController = mSubControllers["Idle"];
	mCurrentController->Synchronize(mCharacter, 0);

	mTargetPositions = mCurrentController->GetCurrentRefPositions();
	Eigen::VectorXd vel(mTargetPositions.rows());
	vel.setZero();
	
	skel->setPositions(mTargetPositions);
	skel->setVelocities(vel);
	skel->computeForwardKinematics(true,true,false);

	mIsWaiting = false;

}
void MetaController::Step()
{
	mCurrentController->Step(mWorld, mCharacter);
	mTargetPositions = mCurrentController->GetCurrentRefPositions();
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mTotalSteps += 1;

	if(mIsWaiting && mCurrentController->Synchronizable(mWaiting.first)) {
		mCurrentController = mSubControllers[mWaiting.first];
		mCurrentController->Synchronize(mCharacter, mWaiting.second);
		mIsWaiting = false;
	}
}
void MetaController::SwitchController(std::string type, int frame)
{
	mIsWaiting = true;
	mWaiting = std::pair<std::string, double>(type, frame);
}
} //end of namespace DPhy
