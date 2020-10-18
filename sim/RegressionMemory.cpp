#include "RegressionMemory.h"
#include <iostream>
#include <fstream>
namespace DPhy
{
bool
IsEqualParam(Param p0, Param p1) {
	if((p0.param_normalized - p1.param_normalized).norm() > 1e-8)
		return false;
	if(abs(p0.reward - p1.reward) > 1e-8)
		return false;
	for(int i = 0; i < p0.cps.size(); i++) {
		if((p0.cps[i] - p1.cps[i]).norm() > 1e-8)
			return false;
	}

	return true;
}
RegressionMemory::
RegressionMemory() :mRD(), mMT(mRD()), mUniform(0.0, 1.0) {
	mNumActivatedPrev = 0;
}
void
RegressionMemory::
InitParamSpace(Eigen::VectorXd paramBase, std::pair<Eigen::VectorXd, Eigen::VectorXd> paramSpace, Eigen::VectorXd paramUnit, 
	double nDOF, double nknots) {
	mDim = paramBase.rows();
	mDimDOF = nDOF;
	mNumKnots = nknots;

	mParamScale.resize(mDim);
	for(int i =0 ; i < mDim; i++) {
		mParamScale(i) = 1.0 / (paramSpace.second(i) - paramSpace.first(i));
	}

	mParamMin = paramSpace.first;
	mParamMax = paramSpace.second;

	mParamGridUnit = paramUnit;
	mRadiusNeighbor = 0.01;
	mThresholdActivate = 5;
	mThresholdUpdate = mDim;

	Eigen::VectorXd base(mDim);
	base.setZero();
	mParamDeactivated.insert(std::pair<Eigen::VectorXd, int>(base, 1));
	for(int i = 0; i < mDim; i++) {
		std::vector<Eigen::VectorXd> vecs;
		double range = std::floor(1.0 / mParamGridUnit(i)) + 1;
		int j = 1;
		while(j < range) {
			auto iter = mParamDeactivated.begin();
			while(iter != mParamDeactivated.end()) {
				Eigen::VectorXd p = iter->first;
				p(i) = j;
				vecs.push_back(p);
				iter++;
			}
			j += 1;
		}
		for(int j = 0; j < vecs.size(); j++) {
			mParamDeactivated.insert(std::pair<Eigen::VectorXd, int>(vecs[j], 1));
		}	
	}
}
std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<double>>
RegressionMemory::
GetTrainingData() {
	std::vector<Eigen::VectorXd> x;
	std::vector<Eigen::VectorXd> y;
	std::vector<double> r;

	auto iter = mGridMap.begin();
	while(iter != mGridMap.end()) {
		std::vector<Param> p = iter->second->GetParams();
		for(int i = 0; i < p.size(); i++) {
			for(int j = 0; j < mNumKnots; j++) {
				Eigen::VectorXd x_elem(mDim + 1);
				x_elem << j, p[i].param_normalized;
				x.push_back(x_elem);
				y.push_back((p[i].cps)[j]);
			}
			r.push_back(p[i].reward);
		} 
		iter++;
	}
	return std::tuple<std::vector<Eigen::VectorXd>, 
					  std::vector<Eigen::VectorXd>, 
					  std::vector<double>> (x, y, r);
}
void
RegressionMemory::
SaveParamSpace(std::string path) {
	auto x_y_r = GetTrainingData();
	std::vector<Eigen::VectorXd> x = std::get<0>(x_y_r);
	std::vector<Eigen::VectorXd> y = std::get<1>(x_y_r);
	std::vector<double> r = std::get<2>(x_y_r);

	std::ofstream ofs(path);
	
	ofs << mDim << std::endl;
	ofs << mDimDOF << std::endl;
	ofs << mNumKnots << std::endl;

	ofs << mThresholdActivate << std::endl;
	ofs << mThresholdUpdate << std::endl;
	ofs << mNumActivatedPrev << std::endl;
	ofs << mRadiusNeighbor << std::endl;

	ofs << mParamScale.transpose() << std::endl;
	ofs << mParamMin.transpose() << std::endl;
	ofs << mParamMax.transpose() << std::endl;
	ofs << mParamGridUnit.transpose() << std::endl;
	ofs << mParamGoalCur.transpose() << std::endl;

	int count = 0;
	for(int i = 0; i < r.size(); i++) {
		ofs << r[i] << std::endl;
		for(int j = 0; j < mNumKnots; j++) {
			ofs << x[count].transpose() << ", " << y[count].transpose() << std::endl;
			count += 1;
		}
	}
	ofs.close();


}
void
RegressionMemory::
LoadParamSpace(std::string path) {
	char buffer[256];

	std::ifstream is;
	is.open(path);

	if(is.fail())
		return;

	is >> buffer;
	mDim = atoi(buffer);

	is >> buffer;
	mDimDOF = atoi(buffer);

	is >> buffer;
	mNumKnots = atoi(buffer);

	is >> buffer;
	mThresholdActivate = atoi(buffer);
	is >> buffer;
	mThresholdUpdate = atoi(buffer);
	is >> buffer;
	mNumActivatedPrev = atoi(buffer);
	is >> buffer;
	mRadiusNeighbor = atof(buffer);

	mParamScale.resize(mDim);
	for(int i = 0; i < mDim; i++) 
	{
		is >> buffer;
		mParamScale[i] = atof(buffer);

	}

	mParamMin.resize(mDim);
	for(int i = 0; i < mDim; i++) 
	{
		is >> buffer;
		mParamMin[i] = atof(buffer);

	}

	mParamMax.resize(mDim);
	for(int i = 0; i < mDim; i++) 
	{
		is >> buffer;
		mParamMax[i] = atof(buffer);

	}

	mParamGridUnit.resize(mDim);
	for(int i = 0; i < mDim; i++) 
	{
		is >> buffer;
		mParamGridUnit[i] = atof(buffer);

	}

	mParamGoalCur.resize(mDim);
	for(int i = 0; i < mDim; i++) 
	{
		is >> buffer;
		mParamGoalCur[i] = atof(buffer);

	}

	mParamDeactivated.clear();
	
	Eigen::VectorXd base(mDim);
	base.setZero();
	mParamDeactivated.insert(std::pair<Eigen::VectorXd, int>(base, 1));
	for(int i = 0; i < mDim; i++) {
		std::vector<Eigen::VectorXd> vecs;
		double range = std::floor(1.0 / mParamGridUnit(i)) + 1;
		int j = 1;
		while(j < range) {
			auto iter = mParamDeactivated.begin();
			while(iter != mParamDeactivated.end()) {
				Eigen::VectorXd p = iter->first;
				p(i) = j;
				vecs.push_back(p);
				iter++;
			}
			j += 1;
		}
		for(int j = 0; j < vecs.size(); j++) {
			mParamDeactivated.insert(std::pair<Eigen::VectorXd, int>(vecs[j], 1));
		}	
	}

	while(!is.eof()) {
		//reward 
		is >> buffer;
		double reward = atof(buffer);

		Eigen::VectorXd param(mDim);
		std::vector<Eigen::VectorXd> cps;
		for(int i = 0; i < mNumKnots; i++) {
			is >> buffer;
			for(int j = 0; j < mDim; j++) 
			{
				is >> buffer;
				param[j] = atof(buffer);
			}
			// comma
			is >> buffer;

			Eigen::VectorXd cp(mDimDOF);
			for(int j = 0; j < mDimDOF; j++) 
			{
				is >> buffer;
				cp[j] = atof(buffer);

			}
		}

		Param p;
		p.param_normalized = param;
		p.cps = cps;
		p.reward = reward;
		AddMapping(p);
	}
	is.close();
	mNumActivatedPrev = mParamActivated.size();

}
Eigen::VectorXd
RegressionMemory::
GetNearestPointOnGrid(Eigen::VectorXd p) {
	Eigen::VectorXd nearest(mDim);
	for(int i = 0; i < mDim; i++) {
		double c =  p(i) / mParamGridUnit(i);
		double c_floor =  std::floor(c) ;
		if(c - c_floor > 0.5) {
			nearest(i) = c_floor + 1;
		} else 
			nearest(i) = c_floor;
	}
	return nearest;
}
std::vector<Eigen::VectorXd> 
RegressionMemory::
GetNeighborPointsOnGrid(Eigen::VectorXd p,  double radius) {
	Eigen::VectorXd nearest = GetNearestPointOnGrid(p);
	return GetNeighborPointsOnGrid(p, nearest, radius);
}
std::vector<Eigen::VectorXd> 
RegressionMemory::
GetNeighborPointsOnGrid(Eigen::VectorXd p, Eigen::VectorXd nearest, double radius) {
	Eigen::VectorXd range(mDim);
	range.setZero();

	for(int i = 0; i < mDim; i++) {
		if(p(i) + radius > 0.5 * mParamGridUnit(i)) {
			range(i) = 1;
		} else if(p(i) - radius < -0.5 * mParamGridUnit(i)) {
			range(i) = -1;
		}
	}
	std::vector<Eigen::VectorXd> neighborlist;
	neighborlist.push_back(nearest);
	for(int i = 0; i < mDim; i++) {
		if(range(i) != 0) {
			std::vector<Eigen::VectorXd> vecs;
			for(int j = 0; j < neighborlist.size(); j++) {
				Eigen::VectorXd n = neighborlist[j];
				n(i) += range(i);
				vecs.push_back(n);
			}
			for(int j = 0; j < vecs.size(); j++) {
				neighborlist.push_back(vecs[j]);
			}
		}
	}
	return neighborlist;
}
void 
RegressionMemory::
AddMapping(Param p) {
	Eigen::VectorXd nearest = GetNearestPointOnGrid(p.param_normalized);
	AddMapping(nearest, p);
}
void 
RegressionMemory::
AddMapping(Eigen::VectorXd nearest, Param p) {
	auto iter = mGridMap.find(nearest);
	if (iter != mGridMap.end()) {
		ParamCube* pcube = iter->second;
		pcube->PutParam(p);
		if(!pcube->GetActivated() && (pcube->GetNumParams() > mThresholdActivate)) {
			pcube->SetActivated(true);
			mParamActivated.insert(std::pair<Eigen::VectorXd, int>(nearest, 1));
			mParamDeactivated.erase(nearest);
		}
	} else {
		ParamCube* pcube = new ParamCube(nearest);
		pcube->PutParam(p);
		mGridMap.insert(std::pair<Eigen::VectorXd, ParamCube* >(nearest, pcube));
	}
}
double 
RegressionMemory::
GetDistanceNorm(Eigen::VectorXd p0, Eigen::VectorXd p1) {
	double r = 0;
	for(int i = 0; i < mDim; i++) {
		r += pow((p0(i) - p1(i)), 2) / pow(mParamGridUnit(i), 2);
	}
	return std::sqrt(r);
}
Eigen::VectorXd 
RegressionMemory::
Normalize(Eigen::VectorXd p) {
	return (p - mParamMin) * mParamScale;
}
void
RegressionMemory::
DeleteMappings(Eigen::VectorXd nearest, std::vector<Param> ps) {

	auto iter = mGridMap.find(nearest);
	if (iter != mGridMap.end()) {
		ParamCube* pcube = iter->second;
		std::vector<Param> ps_new;
		std::vector<Param> ps_old = pcube->GetParams();
		int count = 0;
		for(int i = 0; i < ps_old.size(); i++) {
			if(IsEqualParam(ps_old[i], ps[count])) {
				count += 1;
			} else {
				ps_new.push_back(ps_old[i]);
			}
		}
		pcube->PutParams(ps_new);
		bool wasActivated = pcube->GetActivated();
		if(wasActivated && ps_new.size() <= mThresholdActivate) {
			pcube->SetActivated(false);
			mParamActivated.erase(nearest);
			mParamDeactivated.insert(std::pair<Eigen::VectorXd, int>(nearest, 1));
		}
	} 
}
Eigen::VectorXd 
RegressionMemory::
UniformSample() {
	while(1) {
		double r = mUniform(mMT);
		r = std::floor(r * mParamActivated.size());
		auto it = std::next(mParamActivated.begin(),(int)r);
		Eigen::VectorXd idx = it->first; 
		Eigen::VectorXd p(mDim);

		Eigen::VectorXd range(mDim);
		range.setZero();

		for(int i = 0; i < mDim; i++) {
			double x = mUniform(mMT) - 0.5;
			p(i) = idx(i) + x * mParamGridUnit(i);
			if(p(i) > 1 || p(i) < 0) {
				p(i) = std::min(1.0, std::max(0.0, p(i)));
			}

		}
		std::vector<Eigen::VectorXd> checklist = GetNeighborPointsOnGrid(p, idx, 2 * mRadiusNeighbor);
		
		for(int i = 0; i < checklist.size(); i++) {
			auto iter = mGridMap.find(checklist[i]);
			if (iter != mGridMap.end()) {
				std::vector<Param> ps = iter->second->GetParams();
				for(int j = 0; j < ps.size(); j++) {
					if(GetDistanceNorm(ps[j].param_normalized, p) < 2 * mRadiusNeighbor) {
						return p.cwiseProduct(mParamScale) + mParamMin;
					}
				}
			}
		}
	}
}

bool 
RegressionMemory::
UpdateParamSpace(std::tuple<std::vector<Eigen::VectorXd>, Eigen::VectorXd, double> candidate) {
	Eigen::VectorXd candidate_param = std::get<1>(candidate);
	for(int i = 0; i < mDim; i++) {
		if(candidate_param(i) > mParamMax(i) || candidate_param(i) < mParamMin(i)) {
			return false;
		}
	}
	Eigen::VectorXd candidate_scaled = (candidate_param - mParamMin) * mParamScale;
	Eigen::VectorXd nearest = GetNearestPointOnGrid(candidate_scaled);

	std::vector<Eigen::VectorXd> checklist;
	Eigen::VectorXd range(mDim);
	range.setZero();
	checklist.push_back(range);

	for(int i = 0; i < mDim; i++) {
		if((candidate_scaled(i) - nearest(i)) + mRadiusNeighbor * mParamGridUnit(i) > 0.5 * mParamGridUnit(i)) {
			range(i) = 1;
		} else if((candidate_scaled(i) - nearest(i)) - mRadiusNeighbor * mParamGridUnit(i) < -0.5 * mParamGridUnit(i)) {
			range(i) = -1;
		}
	} 
	for(int i = 0; i < mDim; i++) {
		if(range(i) != 0) {
			std::vector<Eigen::VectorXd> vecs;
			for(int j = 0; j < checklist.size(); j++) {
				Eigen::VectorXd ch = checklist[j];
				ch(i) = range(i);
				vecs.push_back(ch);
			}
			for(int j = 0; j < vecs.size(); j++) {
				checklist.push_back(vecs[j]);
			}
		}
	}
	bool flag = true;
	std::vector<std::pair<Eigen::VectorXd, std::vector<Param>>> to_be_deleted;
	for(int i = 0 ; i < checklist.size(); i++) {
		auto iter = mGridMap.find(nearest + checklist[i]);
		if (iter != mGridMap.end()) {
			ParamCube* pcube = iter->second;
			std::vector<Param> ps = pcube->GetParams();
			std::vector<Param> p_delete;
			for(int j =0; j < ps.size(); j++) {
				if(GetDistanceNorm(candidate_scaled, ps[j].param_normalized) < mRadiusNeighbor) {
					if(ps[j].reward < std::get<2>(candidate)) {
						p_delete.push_back(ps[j]);
					} else {
						flag = false;
						break;
					}
				} 
			}
			if(!flag)
				break;
			else if(p_delete.size() != 0) {
				to_be_deleted.push_back(std::pair<Eigen::VectorXd, std::vector<Param>>(nearest + checklist[i], p_delete));
			}
		}
	}
	if(flag) {
		for(int i = 0; i < to_be_deleted.size(); i++) {
			DeleteMappings(to_be_deleted[i].first, to_be_deleted[i].second);
		}

		Param p;
		p.param_normalized = candidate_scaled;
		p.reward = std::get<2>(candidate);
		p.cps = std::get<0>(candidate);
	 
	 	AddMapping(nearest, p);

	}
	return flag;

}
Eigen::VectorXd 
RegressionMemory::
SelectNewParamGoal() { 
	std::vector<std::pair<Eigen::VectorXd, double>> candidate;
	while(candidate.size() < 10) {
		double r = mUniform(mMT);
		r = std::floor(r * mParamDeactivated.size());
		auto it_da = std::next(mParamDeactivated.begin(), (int)r);
		Eigen::VectorXd idx = it_da->first; 
		Eigen::VectorXd p(mDim);

		Eigen::VectorXd range(mDim);
		range.setZero();

		for(int i = 0; i < mDim; i++) {
			double x = mUniform(mMT) - 0.5;
			p(i) = idx(i) + x * mParamGridUnit(i);
			if(p(i) > 1 || p(i) < 0) {
				p(i) = std::min(1.0, std::max(0.0, p(i)));
			} 
		}

		std::vector<Eigen::VectorXd> checklist = GetNeighborPointsOnGrid(p, idx, 2 * mRadiusNeighbor);
		bool flag = true;
		for(int i = 0; i < checklist.size(); i++) {
			auto it_gm = mGridMap.find(idx + checklist[i]);
			if (it_gm != mGridMap.end()) {
				std::vector<Param> ps = it_gm->second->GetParams();
				for(int j = 0; j < ps.size(); j++) {
					if(GetDistanceNorm(ps[j].param_normalized, p) < 2 * mRadiusNeighbor) {
						flag = false;
						break;
					}
				}
				if(!flag)
					break;
			}
		}
		double dist = 1e5;
		if(flag) {
			auto it_a = mParamActivated.begin();
			while (it_a != mParamActivated.end()) {
				double d = GetDistanceNorm(it_a->first, p);
				if(d < dist) {
					dist = d;
				}
				it_a++;
			}
			candidate.push_back(std::pair<Eigen::VectorXd, double>(p, dist));
		}
	}
	Eigen::VectorXd max_p;
	double max_r = 0;
	for(int i = 0; i < candidate.size(); i++) {
		if(candidate[i].second > max_r) {
			max_r = candidate[i].second;
			max_p = candidate[i].first;
		}
	}
	return max_p.cwiseProduct(mParamScale) + mParamMin;

}

bool 
RegressionMemory::
IsSpaceExpanded() { 
	if((mParamActivated.size() - mNumActivatedPrev) > mThresholdUpdate) {
		mNumActivatedPrev = mParamActivated.size();
		return true;
	} 
	return false;
}
bool
RegressionMemory::
IsSpaceFullyExplored() {
	if(mParamDeactivated.size() <= mThresholdUpdate) {
		return true;
	}
	return false;
}
};