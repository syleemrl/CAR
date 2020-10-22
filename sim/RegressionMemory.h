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
};
class ParamCube
{
public:
	ParamCube(Eigen::VectorXd i) { idx = i; activated = false; }
	Eigen::VectorXd GetIdx(){ return idx; }
	void PutParam(Param p) { param.push_back(p); }
	int GetNumParams() { return param.size(); }
	std::vector<Param> GetParams() {return param;}
	void PutParams(std::vector<Param> ps) { param = ps;}
	void SetActivated(bool ac) { activated = ac; }
	bool GetActivated() { return activated;}
private:
	Eigen::VectorXd idx;
	std::vector<Param> param;
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

	Eigen::VectorXd UniformSample(int n=2);
	bool UpdateParamSpace(std::tuple<std::vector<Eigen::VectorXd>, Eigen::VectorXd, double> candidate);
	Eigen::VectorXd SelectNewParamGoal();
	std::vector<std::pair<Eigen::VectorXd, std::vector<Eigen::VectorXd>>> SelectNewParamGoalCandidate();

	void AddMapping(Param p);
	void AddMapping(Eigen::VectorXd nearest, Param p);
	void DeleteMappings(Eigen::VectorXd nearest, std::vector<Param> ps);

	double GetDistanceNorm(Eigen::VectorXd p0, Eigen::VectorXd p1);	
	Eigen::VectorXd GetNearestPointOnGrid(Eigen::VectorXd p);
	Eigen::VectorXd GetNearestActivatedParam(Eigen::VectorXd p);
	std::vector<Eigen::VectorXd> GetNeighborPointsOnGrid(Eigen::VectorXd p, double radius);
	std::vector<Eigen::VectorXd> GetNeighborPointsOnGrid(Eigen::VectorXd p, Eigen::VectorXd nearest, double radius);
	std::vector<Eigen::VectorXd> GetNeighborParams(Eigen::VectorXd p);
	std::vector<std::pair<double, Param>> GetNearestParams(Eigen::VectorXd p, int n);

	Eigen::VectorXd Normalize(Eigen::VectorXd p);
	Eigen::VectorXd Denormalize(Eigen::VectorXd p);
	void SaveContinuousParamSpace(std::string path);

	bool IsSpaceExpanded();
	bool IsSpaceFullyExplored();

	Eigen::VectorXd GetParamGoal() {return mParamGoalCur; }
	void SetParamGoal(Eigen::VectorXd paramGoal) { mParamGoalCur = paramGoal; }
	void SetRadius(double rn) { mRadiusNeighbor = rn; }
	void SetParamGridUnit(Eigen::VectorXd gridUnit) { mParamGridUnit = gridUnit;}
	int GetDim() {return mDim; }
	void ResetPrevSpace();
	std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<double>> GetTrainingData();
	int GetTimeFromLastUpdate() { return mTimeFromLastUpdate; }

	double GetParamReward(Eigen::VectorXd p, Eigen::VectorXd p_goal);
	std::vector<Eigen::VectorXd> GetCPSFromNearestParams(Eigen::VectorXd p_goal);

private:
	std::map<Eigen::VectorXd, int> mParamActivated;
	std::map<Eigen::VectorXd, int> mParamDeactivated;

	Eigen::VectorXd mParamScale;
	Eigen::VectorXd mParamScaleInv;
	Eigen::VectorXd mParamGoalCur;
	Eigen::VectorXd mParamMin;
	Eigen::VectorXd mParamMax;
	Eigen::VectorXd mParamGridUnit;

	std::map< Eigen::VectorXd, ParamCube* > mGridMap;
	Param mParamBVH;

	double mRadiusNeighbor;
	int mDim;
	int mDimDOF;
	int mNumKnots;
	int mNumActivatedPrev;
	int mThresholdUpdate;
	int mThresholdActivate;
	int mTimeFromLastUpdate;
	int mNumElite;

	std::random_device mRD;
	std::mt19937 mMT;
	std::uniform_real_distribution<double> mUniform;
};
}
#endif