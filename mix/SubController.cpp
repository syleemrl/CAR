#include "SubController.h"

namespace DPhy{

SubController::SubController(CTR_TYPE type, std::string motion, std::string ppo, std::string reg)
: mType(type), mMotion(motion)
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
	        // mParamRange = mReferenceManager->GetParamRange();
	       
	        path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
		//	mRegressionMemory->SaveContinuousParamSpace(path + "param_cspace");
    	}
    	if(ppo != "") {
   //  		this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
			// mController->SetGoalParameters(mReferenceManager->GetParamCur());

    		p::object ppo_main = p::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + ppo;
			// this->mPPO.attr("initRun")(path,
			// 						   this->mController->GetNumState(), 
			// 						   this->mController->GetNumAction());
			// RunPPO();
    	}
    
    } catch (const p::error_already_set&) {
        PyErr_Print();
    }    
}

}