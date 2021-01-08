#include "SubController.h"
#include "MetaController.h"

namespace DPhy{

SubController::SubController(std::string type, MetaController* mc, std::string motion, std::string ppo, std::string reg)
: mType(type), mMotion(motion), mMC(mc)
{
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
    DPhy::Character* mCharacter = new DPhy::Character(path);

    mReferenceManager = new DPhy::ReferenceManager(mCharacter);
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);

    mUseReg= (reg!="");
    if(mUseReg){
    	mRegressionMemory = new DPhy::RegressionMemory();
		mReferenceManager->SetRegressionMemory(mRegressionMemory);
    }

    path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(ppo, '/')[0] + std::string("/");
	mReferenceManager->InitOptimization(1, path, true,mType);
    mReferenceManager->LoadAdaptiveMotion("ref_1");


    Py_Initialize();
    np::initialize();
    try {
    	if(mUseReg) {
			p::object reg_main = p::import("regression");
	        this->mRegression = reg_main.attr("Regression")();
	        std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
	        this->mRegression.attr("initRun")(path, mReferenceManager->GetParamGoal().rows() + 1, mReferenceManager->GetDOF() + 1);
			mRegressionMemory->LoadParamSpace(path + "param_space");
			std::cout << mRegressionMemory->GetVisitedRatio() << std::endl;
	        //not needed // mParamRange = mReferenceManager->GetParamRange();
	       
	        // path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
			//originally commented out also //	mRegressionMemory->SaveContinuousParamSpace(path + "param_cspace");
    	}
    	std::cout<<"reg done"<<std::endl;

    	if(ppo != "") {
    		//not needed // this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
			//not needed // mController->SetGoalParameters(mReferenceManager->GetParamCur());

    		p::object ppo_main = p::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + ppo;

			std::cout<<"mMC statenum: "<<  mMC->GetNumState() <<" / ref : "<<mReferenceManager->GetParamGoal().rows()<<" / total: "<<(  mMC->GetNumState() + mReferenceManager->GetParamGoal().rows())<<std::endl;
			this->mPPO.attr("initRun")(path, 
									   mMC->GetNumState() + mReferenceManager->GetParamGoal().rows(), 
									   mMC->GetNumAction());
			
			std::cout<<"DONE restore ppo (initRun)"<<std::endl;
			// RunPPO();
    	}
    
    } catch (const p::error_already_set&) {
        PyErr_Print();
    }    
}


//////////////////////////////////// FW_JUMP ////////////////////////////////////

FW_JUMP_Controller::FW_JUMP_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg)
: SubController(std::string("FW_JUMP"), mc, motion, ppo, reg)
{}

bool FW_JUMP_Controller::IsTerminalState()
{
	//TODO
	return false;
}

void FW_JUMP_Controller::reset(double frame, double frameOnPhase)
{
	//TODO
	// this->mCurrentFrame = frame;
	// this->mCurrentFrameOnPhase = frameOnPhase;

}

bool FW_JUMP_Controller::Step()
{
	//TODO
	
	// this->mCurrentFrame += mMC->mAdaptiveStep;
	// this->mCurrentFrameOnPhase += mMC->mAdaptiveStep;

}
//////////////////////////////////// WALL_JUMP ////////////////////////////////////

WALL_JUMP_Controller::WALL_JUMP_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg)
: SubController(std::string("WALL_JUMP"), mc, motion, ppo, reg)
{}

bool WALL_JUMP_Controller::IsTerminalState()
{
	//TODO
	return false;
}

void WALL_JUMP_Controller::reset(double frame, double frameOnPhase){
	this->mCurrentFrame = frame;
	this->mCurrentFrameOnPhase = frameOnPhase;

	bool isAdaptive = true;

	if(leftHandConstraint && mCurrentFrameOnPhase <30) removeHandFromBar(true);
	if(rightHandConstraint && mCurrentFrameOnPhase <30) removeHandFromBar(false);

	//45, 59
	left_detached= (mCurrentFrameOnPhase >=37) ? true: false; 
	right_detached= (mCurrentFrameOnPhase >=51) ? true: false;
}

bool WALL_JUMP_Controller::Step()
{
	// this->mCurrentFrame += mMC->mAdaptiveStep;
	// this->mCurrentFrameOnPhase += mMC->mAdaptiveStep;
	this->mCurrentFrameOnPhase = mMC->mCurrentFrameOnPhase;
	
	if(mCurrentFrameOnPhase >=27 && !left_detached && !leftHandConstraint) attachHandToBar(true, Eigen::Vector3d(0.06, -0.025, 0));
	else if(mCurrentFrameOnPhase >=37 && leftHandConstraint) { removeHandFromBar(true); left_detached= true; }

	if(mCurrentFrameOnPhase >=27 && !right_detached && !rightHandConstraint) attachHandToBar(false, Eigen::Vector3d(-0.06, -0.025, 0));
	else if(mCurrentFrameOnPhase >=51 && rightHandConstraint) {removeHandFromBar(false); right_detached =true;}

}

void WALL_JUMP_Controller::attachHandToBar(bool left, Eigen::Vector3d offset){

	std::string hand = (left) ? "LeftHand" : "RightHand";
	dart::dynamics::BodyNodePtr hand_bn = mMC->mCharacter->GetSkeleton()->getBodyNode(hand);
	dart::dynamics::BodyNodePtr bar_bn = this->mCurObject->getBodyNode("Jump_Box");
	Eigen::Vector3d jointPos = hand_bn->getWorldTransform() * offset;

	Eigen::VectorXd mParamGoal = mReferenceManager->GetParamGoal();
	std::cout<<"mParamGoal; "<<mParamGoal.transpose()<<std::endl;
	double obj_height = mParamGoal[0];
	Eigen::Vector3d middle = bar_bn->getWorldTransform()*Eigen::Vector3d(0, 0.45, 0);
	Eigen::Vector2d diff_middle (jointPos[1]-middle[1], jointPos[2]-middle[2]);
	double distance = diff_middle.norm();

	std::cout<<mCurrentFrameOnPhase<<", attach, "<<left<<": "<<distance<<"/ joint:"<<jointPos.transpose()<<"/ middle:"<<middle.transpose()<<std::endl;

	if(distance > 0.1 || jointPos[2] < (middle[2]-0.1) || jointPos[2] > (middle[2]+0.1) || jointPos[1] > (obj_height+0.05) ) return;

	// mParamCur[0]= mParamGoal[0];

	if(left && leftHandConstraint) removeHandFromBar(true);
	else if(!left && rightHandConstraint) removeHandFromBar(false);

	// if(left) dbg_LeftConstraintPoint = jointPos;
	// else dbg_RightConstraintPoint = jointPos;

	dart::constraint::BallJointConstraintPtr cl = std::make_shared<dart::constraint::BallJointConstraint>( hand_bn, bar_bn, jointPos);
	mMC->mWorld->getConstraintSolver()->addConstraint(cl);

	if(left) leftHandConstraint = cl;
	else rightHandConstraint = cl;

	// if(mRecord){
		std::cout<<"attach "<<mCurrentFrameOnPhase<<" ";
		if(left) std::cout<<"left : ";
		else std::cout<<"right : ";
		std::cout<<jointPos.transpose()<<" distance :"<<distance<<std::endl;
	// }

}


void WALL_JUMP_Controller::removeHandFromBar(bool left){
	// std::cout<<"REMOVE "<<left<<std::endl;
	if(left && leftHandConstraint) {
	    mMC->mWorld->getConstraintSolver()->removeConstraint(leftHandConstraint);
	    leftHandConstraint = nullptr;
    	// dbg_LeftConstraintPoint = Eigen::Vector3d::Zero();

	}else if(!left && rightHandConstraint){
	    mMC->mWorld->getConstraintSolver()->removeConstraint(rightHandConstraint);
    	rightHandConstraint = nullptr;
		// dbg_RightConstraintPoint = Eigen::Vector3d::Zero();	    	
	}

	std::cout<<"remove "<<mCurrentFrameOnPhase<<" ";
	if(left) std::cout<<"left : "<<std::endl;
	else std::cout<<"right : "<<std::endl;
}



} // end namespace DPhy