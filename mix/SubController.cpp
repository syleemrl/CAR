#include "SubController.h"
#include "MetaController.h"

namespace DPhy{

SubController::SubController(std::string type, MetaController* mc, std::string motion, std::string ppo, std::string reg, bool isParametric)
: mType(type), mMotion(motion), mMC(mc), mIsParametric(isParametric)
{
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
    DPhy::Character* mCharacter = new DPhy::Character(path);

    mReferenceManager = new DPhy::ReferenceManager(mCharacter);
    
	mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);

    mUseReg= (reg!="");

    path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(ppo, '/')[0] + std::string("/");
    if(mUseReg){
    	mRegressionMemory = new DPhy::RegressionMemory();
		mReferenceManager->SetRegressionMemory(mRegressionMemory);
		mReferenceManager->InitOptimization(1, path, true,mType);
    }
    else{
    	mReferenceManager->InitOptimization(1, path, false, mType);
    }

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

FW_JUMP_Controller::FW_JUMP_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg, bool isParametric)
: SubController(std::string("FW_JUMP"), mc, motion, ppo, reg, isParametric)
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

WALL_JUMP_Controller::WALL_JUMP_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg, bool isParametric)
: SubController(std::string("WALL_JUMP"), mc, motion, ppo, reg, isParametric)
{}

bool WALL_JUMP_Controller::IsTerminalState()
{
	//TODO
	return false;
}

void WALL_JUMP_Controller::reset(double frame, double frameOnPhase){
	this->mCurrentFrame = frame;
	this->mCurrentFrameOnPhase = frameOnPhase;

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
	if(left_detached && mMC->mCurrentFrameOnPhase<=1) left_detached = false;
	if(right_detached && mMC->mCurrentFrameOnPhase<=1) right_detached = false;

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



//////////////////////////////////// RUN_SWING ////////////////////////////////////

RUN_SWING_Controller::RUN_SWING_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg, bool isParametric)
: SubController(std::string("RUN_SWING"), mc, motion, ppo, reg, isParametric)
{}

bool RUN_SWING_Controller::IsTerminalState()
{
	//TODO
	return false;
}

void RUN_SWING_Controller::reset(double frame, double frameOnPhase){
	this->mCurrentFrame = frame;
	this->mCurrentFrameOnPhase = frameOnPhase;

	if(leftHandConstraint) removeHandFromBar(true);
	if(rightHandConstraint) removeHandFromBar(false);

	//45, 59
	left_detached= (mCurrentFrameOnPhase >=51) ? true: false; 
	right_detached= (mCurrentFrameOnPhase >=51) ? true: false;

	std::cout<<"Reset / "<<left_detached<<" / "<<right_detached<<" / "<<leftHandConstraint<<" / "<<rightHandConstraint<<" / "<<mCurrentFrame<<" / "<<mCurrentFrameOnPhase<<std::endl;

}

bool RUN_SWING_Controller::Step()
{

	if(left_detached && mMC->mCurrentFrameOnPhase<=1) left_detached = false;
	if(right_detached && mMC->mCurrentFrameOnPhase<=1) right_detached = false;

	this->mCurrentFrameOnPhase = mMC->mCurrentFrameOnPhase;
	// [23, 51)
	if(mCurrentFrameOnPhase >=24 && !left_detached && !leftHandConstraint) attachHandToBar(true, Eigen::Vector3d(0.03, -0.025, 0));
	else if(mCurrentFrameOnPhase >=51 && leftHandConstraint) { removeHandFromBar(true); left_detached= true; }
	
	if(mCurrentFrameOnPhase >=24 && !right_detached && !rightHandConstraint) attachHandToBar(false, Eigen::Vector3d(-0.03, -0.025, 0));
	else if(mCurrentFrameOnPhase >=51 && rightHandConstraint) {removeHandFromBar(false); right_detached =true;}



}


void RUN_SWING_Controller::attachHandToBar(bool left, Eigen::Vector3d offset){
	// std::cout<<"attach; "<<left;
	
	std::string hand = (left) ? "LeftHand" : "RightHand";
	dart::dynamics::BodyNodePtr hand_bn = mMC->mCharacter->GetSkeleton()->getBodyNode(hand);
	dart::dynamics::BodyNodePtr bar_bn = this->mCurObject->getBodyNode("Bar");
	Eigen::Vector3d jointPos = hand_bn->getTransform() * offset;

	Eigen::Vector3d bar_pos = bar_bn->getWorldTransform().translation();
	Eigen::Vector3d diff = jointPos- bar_pos; 
	diff[0]=0;
	double distance= diff.norm();

	// std::cout<<", attempt/ distance: "<<distance<<", bar_pos:"<<bar_pos.transpose()<<std::endl;

	if(distance > 0.09) return;

	if(left && leftHandConstraint) removeHandFromBar(true);
	else if(!left && rightHandConstraint) removeHandFromBar(false);

	// if(left) dbg_LeftConstraintPoint = jointPos;
	// else dbg_RightConstraintPoint = jointPos;

	hand_bn->setCollidable(false);

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


void RUN_SWING_Controller::removeHandFromBar(bool left){
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

	std::string hand = (left) ? "LeftHand" : "RightHand";
	dart::dynamics::BodyNodePtr hand_bn = mMC->mCharacter->GetSkeleton()->getBodyNode(hand);
	hand_bn->setCollidable(true);

	// if(mRecord) 
		std::cout<<mMC->mCharacter->GetSkeleton()->getBodyNode("LeftHand")->isCollidable()<<" / "<<mMC->mCharacter->GetSkeleton()->getBodyNode("RightHand")->isCollidable()<<std::endl;
	
	// if(mRecord){
		std::cout<<"remove "<<mCurrentFrameOnPhase<<" ";
		if(left) std::cout<<"left "<<std::endl;
		else std::cout<<"right "<<std::endl;		
	// }
}

//////////////////////////////////// RUN_CONNECT ////////////////////////////////////

RUN_CONNECT_Controller::RUN_CONNECT_Controller(MetaController* mc, std::string motion, std::string ppo, std::string reg, bool isParametric)
: SubController(std::string("RUN_CONNECT"), mc, motion, ppo, reg, isParametric)
{}

bool RUN_CONNECT_Controller::IsTerminalState()
{
	//TODO
	return false;
}

void RUN_CONNECT_Controller::reset(double frame, double frameOnPhase)
{
	//TODO
}

bool RUN_CONNECT_Controller::Step()
{
	//TODO
}

} // end namespace DPhy