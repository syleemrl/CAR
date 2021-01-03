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
	mReferenceManager->InitOptimization(1, path, true);
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

bool FW_JUMP_Controller::Step()
{
	//TODO
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

bool WALL_JUMP_Controller::Step()
{
	//TODO
}




} // end namespace DPhy