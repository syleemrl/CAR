#include "SubController.h"

namespace DPhy
{

class MetaController
{
public:
	MetaController();
	
	void addSubController(SubController* new_sc){mSubControllers[new_sc->mType]= new_sc;}
	std::map<CTR_TYPE, SubController*> mSubControllers;
	std::vector<std::tuple<CTR_TYPE, int, CTR_TYPE, int>> transition_rules;

	SubController* mCurrentController;

	void switchController(CTR_TYPE type, int frame=-1);


	//Controller stuff
	dart::simulation::WorldPtr mWorld;

	int mControlHz;
	int mSimulationHz;
	int mSimPerCon;

	Character* mCharacter;
	std::vector<Character*> mObjects;
	dart::dynamics::SkeletonPtr mGround;

	int mCurrentFrame;
	bool mIsTerminal;


	void SetAction(const Eigen::VectorXd& action);
	int GetNumAction(){return mNumAction;}
	Eigen::VectorXd GetState();
	void Step();


	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;

	Eigen::VectorXd mPDTargetPositions;
	Eigen::VectorXd mPDTargetVelocities;

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
	std::vector<double> mRecordPhase;

	int mNumState, mNumAction;
	int terminationReason;

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
