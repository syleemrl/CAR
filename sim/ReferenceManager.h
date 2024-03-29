#ifndef __DEEP_PHYSICS_REFERENCE_MANAGER_H__
#define __DEEP_PHYSICS_REFERENCE_MANAGER_H__

#include "Functions.h"
#include "Character.h"
#include "CharacterConfigurations.h"
#include "BVH.h"
#include "MultilevelSpline.h"
#include "RegressionMemory.h"
#include <tuple>
#include <mutex>

namespace DPhy
{
struct Fitness
{
	double sum_contact;
	double sum_pos;
	double sum_pos_threshold;
	double sum_vel;
	double sum_vel_threshold;
	double sum_reward;
	double sum_slide;
};
class Motion
{
public:
	Motion(Motion* m) {
		position = m->position;
		velocity = m->velocity;
	}
	Motion(Eigen::VectorXd pos) {
		position = pos;
	}
	Motion(Eigen::VectorXd pos, Eigen::VectorXd vel) {
		position = pos;
		velocity = vel;
	}
	void SetPosition(Eigen::VectorXd pos) { position = pos; }
	void SetVelocity(Eigen::VectorXd vel) { velocity = vel; }

	Eigen::VectorXd GetPosition() { return position; }
	Eigen::VectorXd GetVelocity() { return velocity; }

protected:
	Eigen::VectorXd position;
	Eigen::VectorXd velocity;

};
class ReferenceManager
{
public:
	ReferenceManager(Character* character=nullptr);
	void SaveAdaptiveMotion(std::string postfix="");
	void LoadAdaptiveMotion(std::vector<Eigen::VectorXd> cps);
	void LoadAdaptiveMotion(std::string postfix="");
	void LoadMotionFromBVH(std::string filename);
	void GenerateMotionsFromSinglePhase(int frames, bool blend, std::vector<Motion*>& p_phase, std::vector<Motion*>& p_gen);
	Motion* GetMotion(double t, bool adaptive=false);
	std::vector<Eigen::VectorXd> GetVelocityFromPositions(std::vector<Eigen::VectorXd> pos); 
	Eigen::VectorXd GetPosition(double t, bool adaptive=false);
	int GetPhaseLength() {return mPhaseLength; }
	double GetTimeStep(double t, bool adaptive);

	void SaveTrajectories(std::vector<std::pair<Eigen::VectorXd,double>> data_raw, std::tuple<double, double, Fitness> rewards, Eigen::VectorXd parameters);
	void InitOptimization(int nslaves, std::string save_path, bool adaptive=false);
	void AddDisplacementToBVH(std::vector<Eigen::VectorXd> displacement, std::vector<Eigen::VectorXd>& position);
	void GetDisplacementWithBVH(std::vector<std::pair<Eigen::VectorXd, double>> position, std::vector<std::pair<Eigen::VectorXd, double>>& displacement);
	std::vector<double> GetContacts(double t);
	int GetDOF() {return mDOF; }
	int GetNumCPS() {return mPhaseLength;}

	Eigen::VectorXd GetParamGoal() {return mParamGoal; }
	Eigen::VectorXd GetParamCur() {return mParamCur; }
	std::pair<Eigen::VectorXd, Eigen::VectorXd> GetParamRange() {return std::pair<Eigen::VectorXd, Eigen::VectorXd>(mParamBase, mParamEnd); }
	void SetParamGoal(Eigen::VectorXd g) { mParamGoal = g; }
	void ResetOptimizationParameters(bool reset_displacement=true);
	void SetRegressionMemory(RegressionMemory* r) {mRegressionMemory = r; }
	void SetCPSreg(std::vector<Eigen::VectorXd> cps) {mCPS_reg = cps; }
	void SetCPSexp(std::vector<Eigen::VectorXd> cps) {mCPS_exp = cps; }
	std::vector<Eigen::VectorXd> GetCPSreg() { return mCPS_reg; }
	std::vector<Eigen::VectorXd> GetCPSexp() { return mCPS_exp; }
	void SelectReference();
	std::vector<std::string> GetHierarchyStr() {return mHierarchyStr; }

protected:
	Character* mCharacter;
	double mTimeStep;
	int mBlendingInterval;
	int mPhaseLength;
	std::vector<bool> mFootSliding;
	std::vector<Motion*> mMotions_raw;
	std::vector<Motion*> mMotions_phase;
	std::vector<Motion*> mMotions_phase_adaptive;
	std::vector<std::vector<bool>> mContacts;
	std::vector<Motion*> mMotions_gen;
	std::vector<Motion*> mMotions_gen_adaptive;
	std::vector<std::vector<Motion*>> mMotions_gen_temp;
	std::vector<double> mTimeStep_adaptive;
	
	std::vector<Eigen::VectorXd> mCPS_reg;
	std::vector<Eigen::VectorXd> mCPS_exp;

	double mSlaves;
	std::mutex mLock;

	std::string mPath;
	
	bool isParametric;
	int mDOF;

	Eigen::VectorXd mParamGoal;
	Eigen::VectorXd mParamCur;
	Eigen::VectorXd mParamBase;
	Eigen::VectorXd mParamEnd;

	RegressionMemory* mRegressionMemory;
	
	double mMeanTrackingReward;
	double mMeanParamReward;

	double mThresholdTracking;

	std::random_device mRD;
	std::mt19937 mMT;
	std::uniform_real_distribution<double> mUniform;
	std::vector<std::string> mHierarchyStr;

};
}

#endif