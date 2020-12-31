#include "Character.h"
#include "MultilevelSpline.h"
#include <boost/filesystem.hpp>
#include <Eigen/QR>
#include <fstream>
#include <numeric>
#include <algorithm>

namespace DPhy
{
enum CTR_TYPE{
	FW_JUMP,
	WALL_JUMP,
	RUN_SWING,
	BOX_CLIMB
};

class MetaController;

class SubController
{
public:
	SubController();

	p::object 						mPPO;
	DPhy::ReferenceManager*			mReferenceManager;
	CTR_TYPE name;

	Eigen::VectorXd mParamGoal;

	bool virtual IsTerminalState(MetaController& mc);
	bool virtual Step(MetaController& mc);
};


class MetaController
{
public:
	MetaController();
	
	void addSubController(SubController new_sc){mSubControllers[new_sc.name]= new_sc;}
	std::map<CTR_TYPE, SubController> mSubControllers;
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

	int mCurrentFrame;
	bool mIsTerminal;
};
}
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
