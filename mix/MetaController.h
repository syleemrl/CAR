#ifndef __META_CONTROLLER_H
#define __META_CONTROLLER_H
#include "SubController.h"

namespace DPhy
{

struct Take
{
	Take(std::string ctrl_type):ctrl_type(ctrl_type){}

	std::string ctrl_type;
	Eigen::VectorXd goalParam;
	std::string target_object="";

};

class MetaController
{
public:
	MetaController(std::string ctrl, std::string obj, std::string scenario);
	
	std::map<std::string, SubController*> mSubControllers;
	void loadControllers(std::string ctrl_path);
	void loadScenario(std::string scenario_path);
	void addSubController(SubController* new_sc){mSubControllers[new_sc->mType]= new_sc;}

	SubController* mPrevController;	
	SubController* mCurrentController;
	void switchController(std::string type, int frame=-1);
	void handleTargetObject(int scene_number);
	// void handleTargetObject_init();
	std::map<std::pair<std::string, std::string>, std::pair<int, int>> mTransitionRules;

	void runScenario();

	//World-related stuff
	dart::simulation::WorldPtr mWorld;

	int mControlHz;
	int mSimulationHz;
	int mSimPerCon;

	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;

	void loadSceneObjects(std::string obj_path);
	std::map<std::string, dart::dynamics::SkeletonPtr> mSceneObjects;
	bool mLoadScene= false;


	double mCurrentFrame;
	double mPrevFrame;
	double mPrevFrame2;

	double mCurrentFrameOnPhase;
	double mPrevFrameOnPhase;

	double mTimeElapsed;

	bool mIsTerminal;
	bool mIsNanAtTerminal;

	void reset();
	void SetAction(const Eigen::VectorXd& action);
	int GetNumAction(){return mNumAction;}
	Eigen::VectorXd GetState();
	void Step();
	int GetCurrentFrame(){return mCurrentFrame;}

	int GetNumState();
	bool IsTerminalState(){return mIsTerminal;}
	void UpdateTerminalInfo();
	Eigen::VectorXd GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel);
	void SaveStepInfo();
	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;

	Eigen::VectorXd mPDTargetPositions;
	Eigen::VectorXd mPDTargetVelocities;

	Eigen::VectorXd mPrevTargetPositions;


	Eigen::VectorXd mActions;
	double mAdaptiveStep;

	std::vector<std::string> mInterestedBodies;
	std::vector<std::string> mRewardBodies;
	int mInterestedDof;
	int mRewardDof;

	std::vector<std::string> mEndEffectors;
	std::vector<std::string> mRewardLabels;

	std::vector<Eigen::VectorXd> mRecordPosition;
	std::vector<Eigen::VectorXd> mRecordVelocity;
	std::vector<Eigen::Vector3d> mRecordCOM;
	std::vector<Eigen::VectorXd> mRecordTargetPosition;
	std::vector<Eigen::VectorXd> mRecordBVHPosition;
	std::vector<Eigen::VectorXd> mRecordObjPosition;
	std::vector<std::pair<bool, bool>> mRecordFootContact;
	std::vector<Eigen::VectorXd> mRecordDebug;

	std::vector<double> mRecordPhase;

	int mNumState, mNumAction;
	int terminationReason;

	std::queue<Eigen::VectorXd> mPosQueue;
	std::queue<double> mTimeQueue;

	std::vector<double> mTiming;

	DPhy::ReferenceManager* GetCurrentRefManager(){return mCurrentController->mReferenceManager;}
	DPhy::ReferenceManager* GetPrevRefManager(){return mPrevController->mReferenceManager;}

	int cycle= 0;
	bool scenario_done= false;
	Eigen::Isometry3d mAlign = Eigen::Isometry3d::Identity();

	int mBlendMargin = 3;
	int mCycleStartFrame = 0;


	DPhy::ReferenceManager* mRef1;
	DPhy::ReferenceManager* mRef2;
	double mTime1;
	double mTime2;
	Eigen::Isometry3d mAlign1;
	Eigen::Isometry3d mAlign2;
	double mBlendStep;

	Motion* GetMotion(double t, bool isAdaptive);

	Eigen::Isometry3d calculateAlign(Eigen::Isometry3d align, std::string from, double frame1, std::string to, double frame2);
	int control_mode = 0; 
	// 0: single controller
	// 1: 1->2 (1)
	// 2: 1->2(2)

	std::vector<Take> mTakeList;
	int mCurrentTake;

	std::string m_obj_path;

	int mChangeFrame;
};
} // end namespace DPhy

// 공통: mWorld 유일
	// - setAction은 동일

// 각각: 
	// mCurrentFrameOnPhase, mParamGoal
	// - stepping할때 프레임에 따라 처리해주는 것들 ..
	// - terminal state check

	// transition rules
		// m1 end(f1) -> m2 start(f2), blend yes/no, 
	
	// danceCard (scenario)
	// scene objects 



		// Eigen::VectorXd state = this->mController->GetState();

		// p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
		// np::ndarray na = np::from_object(a);
		// Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

		// this->mController->SetAction(action);
		// this->mController->Step();
		// this->mTiming.push_back(this->mController->GetCurrentFrame());

#endif