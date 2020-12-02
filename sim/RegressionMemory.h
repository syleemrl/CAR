#ifndef __REGMEM_H__
#define __REGMEM_H__
#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <random>

template<>
struct std::less<Eigen::VectorXd>
{ 
	bool operator()(Eigen::VectorXd const& a, Eigen::VectorXd const& b) const {
	    assert(a.size() == b.size());
	    for(size_t i = 0; i < a.size(); i++)
	    {
	        if(a[i] < b[i]) 
	        	return true;
	        if(a[i] > b[i]) 
	        	return false;
	    }
	    return false;
	}
};

namespace DPhy
{
struct Param
{
	Eigen::VectorXd param_normalized;
	std::vector<Eigen::VectorXd> cps;
	double reward;
	bool update;
};
class ParamCube
{
public:
	ParamCube(Eigen::VectorXd i) { idx = i; activated = false; }
	Eigen::VectorXd GetIdx(){ return idx; }
	void PutParam(Param* p) { param.push_back(p); }
	int GetNumParams() { return param.size(); }
	std::vector<Param*> GetParams() {return param;}
	void PutParams(std::vector<Param*> ps);
	void SetActivated(bool ac) { activated = ac; }
	bool GetActivated() { return activated;}
private:
	Eigen::VectorXd idx;
	std::vector<Param*> param;
	bool activated;
};
class RegressionMemory
{
public:
	
	RegressionMemory();
	void InitParamSpace(Eigen::VectorXd paramBvh, std::pair<Eigen::VectorXd, Eigen::VectorXd> paramSpace , Eigen::VectorXd paramUnit, 
						double nDOF, double nknots);
	void SaveParamSpace(std::string path);
	void LoadParamSpace(std::string path);

	std::pair<Eigen::VectorXd , bool> UniformSample(bool visited);
	std::pair<Eigen::VectorXd , bool> UniformSample(double d0, double d1);
	bool UpdateParamSpace(std::tuple<std::vector<Eigen::VectorXd>, Eigen::VectorXd, double> candidate);

	void AddMapping(Param* p);
	void AddMapping(Eigen::VectorXd nearest, Param* p);
	void DeleteMappings(Eigen::VectorXd nearest, std::vector<Param*> ps);
	void UpdateParamState();
	double GetDistanceNorm(Eigen::VectorXd p0, Eigen::VectorXd p1);	
	double GetDensity(Eigen::VectorXd p, bool old=false);
	Eigen::VectorXd GetNearestPointOnGrid(Eigen::VectorXd p);
	Eigen::VectorXd GetNearestActivatedParam(Eigen::VectorXd p);
	std::vector<Eigen::VectorXd> GetNeighborPointsOnGrid(Eigen::VectorXd p, double radius);
	std::vector<Eigen::VectorXd> GetNeighborPointsOnGrid(Eigen::VectorXd p, Eigen::VectorXd nearest, double radius);
	std::vector<Eigen::VectorXd> GetNeighborParams(Eigen::VectorXd p);
	std::vector<std::pair<double, Param*>> GetNearestParams(Eigen::VectorXd p, int n, bool search_neighbor=false, bool old=false, bool inside=false);

	Eigen::VectorXd Normalize(Eigen::VectorXd p);
	Eigen::VectorXd Denormalize(Eigen::VectorXd p);
	void SaveContinuousParamSpace(std::string path);

	double GetVisitedRatio();

	Eigen::VectorXd GetParamGoal() {return mParamGoalCur; }
	void SetParamGoal(Eigen::VectorXd paramGoal);
	void SetRadius(double rn) { mRadiusNeighbor = rn; }
	void SetParamGridUnit(Eigen::VectorXd gridUnit) { mParamGridUnit = gridUnit;}
	int GetDim() {return mDim; }

	std::tuple<std::vector<Eigen::VectorXd>, 
			   std::vector<Eigen::VectorXd>, 
			   std::vector<double>> GetTrainingData();

	double GetParamReward(Eigen::VectorXd p, Eigen::VectorXd p_goal);
	std::vector<Eigen::VectorXd> GetCPSFromNearestParams(Eigen::VectorXd p_goal);
	void SaveLog(std::string path);
	
	std::vector<Param*> mloadAllSamples;

	int GetNumSamples();
	double GetNewSamplesNearGoal() {return mNewSamplesNearGoal;}
	std::tuple<std::vector<Eigen::VectorXd>, 
	   	   std::vector<Eigen::VectorXd>,  
		   std::vector<double>, 
		   std::vector<double>> GetParamSpaceSummary();
	double GetFitness(Eigen::VectorXd p);

private:
	std::map<Eigen::VectorXd, int> mParamActivated;
	std::map<Eigen::VectorXd, int> mParamDeactivated;
	std::map<Eigen::VectorXd, Param*> mParamNew;

	Eigen::VectorXd mParamScale;
	Eigen::VectorXd mParamScaleInv;
	Eigen::VectorXd mParamGoalCur;
	Eigen::VectorXd mParamMin;
	Eigen::VectorXd mParamMax;
	Eigen::VectorXd mParamGridUnit;
	Param* mParamBVH;

	std::map< Eigen::VectorXd, ParamCube* > mGridMap;

	std::vector<std::pair<double, Param*>> mPrevElite;
	std::vector<Eigen::VectorXd> mPrevCPS;   
	double mPrevReward;

	double mRadiusNeighbor;
	double mThresholdInside;
	double mRangeExplore;

	int mDim;
	int mDimDOF;
	int mNumKnots;
	int mNumActivatedPrev;
	int mThresholdUpdate;
	int mThresholdActivate;
	int mNumElite;
	int mNumSamples;

	int mExplorationStep;
	int mNumGoalCandidate;
	int mIdxCandidate;
	std::vector<Eigen::VectorXd> mGoalCandidate;
	std::vector<bool> mGoalExplored;
	std::vector<double> mGoalProgress;
	std::vector<double> mGoalReward;
	std::vector<int> mGoalUpdate;
	std::vector<std::vector<Eigen::VectorXd>> mCPSCandidate;

	std::random_device mRD;
	std::mt19937 mMT;
	std::uniform_real_distribution<double> mUniform;

	std::vector<std::string> mRecordLog;
	double mEliteGoalDistance;
	double mNewSamplesNearGoal;

};
}
#endif