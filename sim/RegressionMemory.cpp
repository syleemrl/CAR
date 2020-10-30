#include "RegressionMemory.h"
#include "Functions.h"
#include <iostream>
#include <fstream>
#include <algorithm>
namespace DPhy
{
void
ParamCube::
PutParams(std::vector<Param*> ps) {
	param = ps;
}
bool
IsEqualParam(Param* p0, Param* p1) {
	if((p0->param_normalized - p1->param_normalized).norm() > 1e-8)
		return false;
	if(abs(p0->reward - p1->reward) > 1e-8)
		return false;
	for(int i = 0; i < p0->cps.size(); i++) {
		if((p0->cps[i] - p1->cps[i]).norm() > 1e-8)
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
InitParamSpace(Eigen::VectorXd paramBvh, std::pair<Eigen::VectorXd, Eigen::VectorXd> paramSpace, Eigen::VectorXd paramUnit, 
	double nDOF, double nknots) {
	mRecordLog.clear();
	mParamNew.clear();

	mNumSamples = 0;
	mDim = paramBvh.rows();
	mDimDOF = nDOF;
	mNumKnots = nknots;

	mParamScale.resize(mDim);
	mParamScaleInv.resize(mDim);

	mParamGridUnit.resize(mDim);
	for(int i = 0 ; i < mDim; i++) {
		if(paramSpace.second(i) == paramSpace.first(i))
			mParamScale(i) = 1.0;			
		else
			mParamScale(i) = 1.0 / (paramSpace.second(i) - paramSpace.first(i));
		mParamGridUnit(i) = paramUnit(i) * mParamScale(i);
		mParamScaleInv(i) = 1.0 / mParamScale(i);
	}

	mParamMin = paramSpace.first;
	mParamMax = paramSpace.second;
	mParamGoalCur = paramBvh;

	mNumElite = 10;
	mRadiusNeighbor = 0.35;
	mThresholdActivate = 5;
	mThresholdUpdate = 2 * mDim;
	mNumGoalCandidate = 5;

	mParamBVH = new Param();
	mParamBVH->cps.clear();
	for(int i = 0; i < mNumKnots; i++) {
		Eigen::VectorXd cps(mDimDOF);
		cps.setZero();
		mParamBVH->cps.push_back(cps);
	}

	mParamBVH->param_normalized = Normalize(paramBvh);
	mParamBVH->reward = 1;
	mGoalInfo.rewards = -1;

	Eigen::VectorXd base(mDim);
	base.setZero();
	mParamDeactivated.insert(std::pair<Eigen::VectorXd, int>(base, 1));
	for(int i = 0; i < mDim; i++) {
		std::vector<Eigen::VectorXd> vecs;
		if(mParamGridUnit(i) == 0 || mParamMin(i) == mParamMax(i))
			continue;
		double range = std::floor(1.0 / mParamGridUnit(i) + 1e-8) + 1;
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

	ResetExploration();

	std::cout << "Regression memory init done: " << std::endl;
	std::cout << "Param min: " << mParamMin.transpose() << std::endl;
	std::cout << "Param max: " << mParamMax.transpose() << std::endl;
	std::cout << "Param bvh normalized: " << mParamBVH->param_normalized.transpose() << std::endl;
	std::cout << "Param unit: " << mParamGridUnit.transpose() << std::endl;
	std::cout << "Param scale: " << mParamScale.transpose() << std::endl;
	std::cout << "Grid size: " << mParamDeactivated.size() << std::endl;
	// auto it = mParamDeactivated.begin();
	// while(it != mParamDeactivated.end()) {
	// 	std::cout << it->first.transpose() << std::endl;
	// }
}
std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<double>>
RegressionMemory::
GetTrainingData(bool update) {
	std::vector<Eigen::VectorXd> x;
	std::vector<Eigen::VectorXd> y;
	std::vector<double> r;
	if(!update) {
		auto iter = mGridMap.begin();
		while(iter != mGridMap.end()) {
			std::vector<Param*> p = iter->second->GetParams();
			for(int i = 0; i < p.size(); i++) {
				for(int j = 0; j < mNumKnots; j++) {
					Eigen::VectorXd x_elem(mDim + 1);
					x_elem << j, p[i]->param_normalized;
					x.push_back(x_elem);
					y.push_back((p[i]->cps)[j]);
				}
				r.push_back(p[i]->reward);
				p[i]->update = false;
			} 
			iter++;
		}
	} else {
		auto iter = mParamNew.begin();
		while(iter != mParamNew.end()) {
			Param* p = iter->second;
			for(int j = 0; j < mNumKnots; j++) {
				Eigen::VectorXd x_elem(mDim + 1);
				x_elem << j, p->param_normalized;
				x.push_back(x_elem);
				y.push_back((p->cps)[j]);
			}
			r.push_back(p->reward);
			p->update = false;
			iter++;
		}
		std::cout << "num new data: " << r.size() << std::endl;
		mParamNew.clear();
	}

	mRecordLog.push_back("save training data: " + std::to_string(r.size()));
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
	
	ofs << mNumActivatedPrev << std::endl;
	ofs << mParamGoalCur.transpose() << std::endl;

	int count = 0;
	for(int i = 0; i < r.size(); i++) {
		ofs << r[i] << std::endl;
		for(int j = 0; j < mNumKnots; j++) {
			ofs << x[count].transpose() << " , " << y[count].transpose() << std::endl;
			count += 1;
		}
	}

	ofs.close();
	std::cout << "save param space : " << x.size() / mNumKnots << std::endl;

	ofs.open(path+"_active");
	auto it = mParamActivated.begin();
	while(it != mParamActivated.end()) {
		ofs << it->first.cwiseProduct(mParamGridUnit).transpose() << std::endl;
		it++;
	}
	ofs.close();

}
bool cmp_pair_int(const std::pair<double, int> &p1, 
		 const std::pair<double, int> &p2){
    if(p1.first > p2.first){
        return true;
    }
    else{
        return false;
    }
}
bool cmp_pair_param(const std::pair<double, Param*> &p1, 
		 const std::pair<double, Param*> &p2){
    if(p1.first > p2.first){
        return true;
    }
    else{
        return false;
    }
}
void
RegressionMemory::
LoadParamSpace(std::string path) {
	mNumSamples = 0;
	mRecordLog.clear();

	char buffer[256];

	std::ifstream is;
	is.open(path);

	if(is.fail())
		return;

	is >> buffer;
	mNumActivatedPrev = atoi(buffer);

	mParamGoalCur.resize(mDim);
	for(int i = 0; i < mDim; i++) 
	{
		is >> buffer;
		mParamGoalCur(i) = atof(buffer);

	}

	mParamDeactivated.clear();
	
	Eigen::VectorXd base(mDim);
	base.setZero();
	mParamDeactivated.insert(std::pair<Eigen::VectorXd, int>(base, 1));
	for(int i = 0; i < mDim; i++) {
		std::vector<Eigen::VectorXd> vecs;
		if(mParamGridUnit(i) == 0 || mParamMin(i) == mParamMax(i))
			continue;
		double range = std::floor(1.0 / mParamGridUnit(i) + 1e-8) + 1;
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

		if(is.eof())
			break;
		
		Eigen::VectorXd param(mDim);
		std::vector<Eigen::VectorXd> cps;
		for(int i = 0; i < mNumKnots; i++) {
			is >> buffer;
			for(int j = 0; j < mDim; j++) 
			{
				is >> buffer;
				param(j) = atof(buffer);
			}
			// comma
			is >> buffer;
			Eigen::VectorXd cp(mDimDOF);
			for(int j = 0; j < mDimDOF; j++) 
			{
				is >> buffer;
				cp(j) = atof(buffer);
			}
			cps.push_back(cp);
		}

		Param* p = new Param();
		p->param_normalized = param;
		p->cps = cps;
		p->reward = reward;
		p->update = false;
		AddMapping(p);
	}

	is.close();
//	mNumActivatedPrev = mParamActivated.size();

	std::cout << "Regression memory load done: " << std::endl;
	std::cout << "Param min: " << mParamMin.transpose() << std::endl;
	std::cout << "Param max: " << mParamMax.transpose() << std::endl;
	std::cout << "Param bvh normalized: " << mParamBVH->param_normalized.transpose() << std::endl;
	std::cout << "Param unit: " << mParamGridUnit.transpose() << std::endl;
	std::cout << "Param scale: " << mParamScale.transpose() << std::endl;
	std::cout << "Param Goal: " << mParamGoalCur.transpose() << std::endl;
	std::cout << "param activated size: " << mParamActivated.size() << std::endl;
	std::cout << "param deactivated size: " << mParamDeactivated.size() << std::endl;
	std::cout << "num samples: " << mNumSamples << std::endl;

	// std::vector<std::pair<int, Eigen::VectorXd>> stats;
	auto iter = mParamActivated.begin();
	while(iter != mParamActivated.end()) {
		auto it = mGridMap.find(iter->first);
		iter++;
	}
	// std::stable_sort(stats.begin(), stats.end(), cmp);
	// for(int i = 0; i < stats.size(); i++) {
	// 	std::cout << stats[i].first << " " << stats[i].second.cwiseProduct(mParamGridUnit).transpose() << std::endl;
	// }
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
Eigen::VectorXd
RegressionMemory::
GetNearestActivatedParam(Eigen::VectorXd p) {
	double dist = 1e5;
	Eigen::VectorXd n;
	auto it = mParamActivated.begin();
	while(it != mParamActivated.end()) {
		Eigen::VectorXd n_param = it->first.cwiseProduct(mParamGridUnit);
		double d = GetDistanceNorm(p, n_param);
		if(d < dist) {
			dist = d;
			n = n_param;
		}
		it++;
	}

	return n;
}
std::vector<Eigen::VectorXd> 
RegressionMemory::
GetNeighborPointsOnGrid(Eigen::VectorXd p, double radius) {
	Eigen::VectorXd nearest = GetNearestPointOnGrid(p);
	return GetNeighborPointsOnGrid(p, nearest, radius);
}
std::vector<Eigen::VectorXd> 
RegressionMemory::
GetNeighborPointsOnGrid(Eigen::VectorXd p, Eigen::VectorXd nearest, double radius) {
	Eigen::VectorXd range(mDim);
	range.setZero();
	Eigen::VectorXd p_n = p - nearest.cwiseProduct(mParamGridUnit);
	for(int i = 0; i < mDim; i++) { 
		if(p_n(i) + radius * mParamGridUnit(i) > 0.5 * mParamGridUnit(i) &&
			p_n(i) - radius * mParamGridUnit(i) < -0.5 * mParamGridUnit(i)) {
			range(i) = 2;
		} if(p_n(i) + radius * mParamGridUnit(i) > 0.5 * mParamGridUnit(i)) {
			range(i) = 1;
		} else if(p_n(i) - radius * mParamGridUnit(i) < -0.5 * mParamGridUnit(i)) {
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
				if(range(i) != 2) {
					n(i) += range(i);
					vecs.push_back(n);
				} else {
					n(i) += 1;
					vecs.push_back(n);

					n(i) += -1;
					vecs.push_back(n);
				}
			}
			for(int j = 0; j < vecs.size(); j++) {
				neighborlist.push_back(vecs[j]);
			}
		}
	}

	return neighborlist;
}
std::vector<Eigen::VectorXd> 
RegressionMemory::
GetNeighborParams(Eigen::VectorXd p) {
	std::vector<Eigen::VectorXd> result;
	std::vector<Eigen::VectorXd> points = GetNeighborPointsOnGrid(p, mRadiusNeighbor * 1.5);
	for(int i = 0; i < points.size(); i++) {
		auto iter = mGridMap.find(points[i]);
		if(iter != mGridMap.end()) {
			std::vector<Param*> ps = iter->second->GetParams();
			for(int j = 0; j < ps.size(); j++) {
				if(GetDistanceNorm(p, ps[j]->param_normalized) <mRadiusNeighbor * 1.5)
					result.push_back(ps[j]->param_normalized);
			}
		}
	}
	return result;
}
std::vector<std::pair<double, Param*>> 
RegressionMemory::
GetNearestParams(Eigen::VectorXd p, int n, bool search_neighbor) {
	std::vector<std::pair<double, Param*>> params;
	if(search_neighbor) {
		std::vector<Eigen::VectorXd> grids = GetNeighborPointsOnGrid(p, 1);
		for(int i = 0; i < grids.size(); i++) {
			auto iter = mGridMap.find(grids[i]);
			if (iter != mGridMap.end()) {
				std::vector<Param*> ps = iter->second->GetParams();
				for(int j = 0; j < ps.size(); j++) {
					params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
				}
			}
		}
	} else {
		auto iter = mGridMap.begin();
		while(iter != mGridMap.end()) {
			std::vector<Param*> ps = iter->second->GetParams();
			for(int j = 0; j < ps.size(); j++) {
				params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
			}
			iter++;
		}
	}
	std::stable_sort(params.begin(), params.end(), cmp_pair_param);
	if(n == -1)
		return params;
	std::vector<std::pair<double, Param*>> result;
	int begin = params.size() - 1;
	int param_end = params.size() - 1 - n;
	int end = std::max(-1, param_end);
	for(int i = begin; i > end; i--) {
		result.push_back(params[i]);
	}
	return result;
}
void 
RegressionMemory::
AddMapping(Param* p) {
	Eigen::VectorXd nearest = GetNearestPointOnGrid(p->param_normalized);
	AddMapping(nearest, p);
}
void 
RegressionMemory::
AddMapping(Eigen::VectorXd nearest, Param* p) {
	auto iter = mGridMap.find(nearest);
	if (iter != mGridMap.end()) {
		ParamCube* pcube = iter->second;

		pcube->PutParam(p);
		if(!pcube->GetActivated() && (pcube->GetNumParams() > mThresholdActivate)) {
			pcube->SetActivated(true);
			mParamActivated.insert(std::pair<Eigen::VectorXd, int>(nearest, 1));
			mParamDeactivated.erase(nearest);
			mRecordLog.push_back("activated: " + vectorXd_to_string(nearest));
		}

	} else {
		ParamCube* pcube = new ParamCube(nearest);
		pcube->PutParam(p);
		mGridMap.insert(std::pair<Eigen::VectorXd, ParamCube* >(nearest, pcube));
	}
	mNumSamples += 1;
}
double 
RegressionMemory::
GetDistanceNorm(Eigen::VectorXd p0, Eigen::VectorXd p1) {
	double r = 0;
	for(int i = 0; i < mDim; i++) {
		if(mParamGridUnit(i) != 0) {
			r += pow((p0(i) - p1(i)), 2) / pow(mParamGridUnit(i), 2);
		}
	}
	return std::sqrt(r);
}
Eigen::VectorXd 
RegressionMemory::
Normalize(Eigen::VectorXd p) {
	return (p - mParamMin).cwiseProduct(mParamScale);
}
Eigen::VectorXd 
RegressionMemory::
Denormalize(Eigen::VectorXd p) {
	return p.cwiseProduct(mParamScaleInv) + mParamMin;
}
void
RegressionMemory::
DeleteMappings(Eigen::VectorXd nearest, std::vector<Param*> ps) {
	auto iter = mGridMap.find(nearest);
	if (iter != mGridMap.end()) {
		ParamCube* pcube = iter->second;

		std::vector<Param*> ps_new;
		std::vector<Param*> ps_old = pcube->GetParams();
		
		int count = 0;
		for(int i = 0; i < ps_old.size(); i++) {
			if(count < ps.size() && IsEqualParam(ps_old[i], ps[count])) {
				delete ps_old[i];
				mNumSamples -= 1;
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
			mRecordLog.push_back("deactivated: " + vectorXd_to_string(nearest));
		}

	} 
}
double 
RegressionMemory::
GetDensity(Eigen::VectorXd p) {
	double density = 0;
	double density_gaussian = 0;

	std::vector<Eigen::VectorXd> neighborlist = GetNeighborPointsOnGrid(p, 1);
	for(int j = 0; j < neighborlist.size(); j++) {
		auto iter = mGridMap.find(neighborlist[j]);
		if(iter != mGridMap.end()) {
			std::vector<Param*> ps = iter->second->GetParams();
			for(int k = 0; k < ps.size(); k++) {
				double d = GetDistanceNorm(p, ps[k]->param_normalized);
				density += 0.1 * std::max(0.0, 1 - d);
				density_gaussian += 0.1 * exp( - pow(d, 2) * 5);
	
			}
		}
	}
	return density_gaussian;

}
Eigen::VectorXd 
RegressionMemory::
UniformSample(int n) {
	while(1) {
		// Eigen::VectorXd p(mDim);
		// for(int j =0 ; j < mDim; j++) {
		// 		p(j) = mUniform(mMT);
		// }
		// if(n == 0)
		// 	return Denormalize(p);
		
		// auto pairs = GetNearestParams(p, 5);
		// double mean = 0;
		// for(int i = 0; i < pairs.size(); i++) {
		// 	mean += pairs[i].first;
		// }
		// mean /= pairs.size();
		// if(mean > 0.6)
		// 	continue;
		// else
		// 	return Denormalize(p);


		// int count = 0;
		// std::vector<Eigen::VectorXd> neighborlist = GetNeighborPointsOnGrid(p, 1.5 * mRadiusNeighbor);
		// for(int j = 0; j < neighborlist.size(); j++) {
		// 	auto iter = mGridMap.find(neighborlist[j]);
		// 	if(iter != mGridMap.end()) {
		// 		std::vector<DPhy::Param> ps = iter->second->GetParams();
		// 		for(int k = 0; k < ps.size(); k++) {
		// 			if(GetDistanceNorm(p, ps[k].param_normalized) < mRadiusNeighbor * 1.5) {
		// 				count += 1;
		// 				if(count >= n)
		// 					return Denormalize(p);
		// 			}
		// 		}
		// 	}

		// }
		double r = mUniform(mMT);
		r = std::floor(r * mParamActivated.size());
		if(r == mParamActivated.size())
			r -= 1;

		auto it = std::next(mParamActivated.begin(),(int)r);
		Eigen::VectorXd idx = it->first; 
		Eigen::VectorXd p(mDim);

		Eigen::VectorXd range(mDim);
		range.setZero();

		for(int i = 0; i < mDim; i++) {
			double x = mUniform(mMT) - 0.5;
			p(i) = (idx(i) + x) * mParamGridUnit(i);
			if(p(i) > 1 || p(i) < 0) {
				p(i) = std::min(1.0, std::max(0.0, p(i)));
			}

		}
		double d = GetDensity(p);
		// auto pairs = GetNearestParams(p, 5, true);
		// if(pairs.size() != 5)
		// 	continue;

		// double mean = 0;
		// for(int i = 0; i < pairs.size(); i++) {
		// 	mean += pairs[i].first;
		// }
		// mean /= pairs.size();

		if(d < 0.4)
			continue;
		else
			return Denormalize(p);

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
	Eigen::VectorXd candidate_scaled = (candidate_param - mParamMin).cwiseProduct(mParamScale);
	Eigen::VectorXd nearest = GetNearestPointOnGrid(candidate_scaled);

	std::vector<Eigen::VectorXd> checklist = GetNeighborPointsOnGrid(candidate_scaled, nearest, mRadiusNeighbor);

	int n_compare = 0;
	bool flag = true;
	std::vector<std::pair<Eigen::VectorXd, std::vector<Param*>>> to_be_deleted;
	for(int i = 0 ; i < checklist.size(); i++) {
		auto iter = mGridMap.find(checklist[i]);
		if (iter != mGridMap.end()) {
			ParamCube* pcube = iter->second;
			std::vector<Param*> ps = pcube->GetParams();
			std::vector<Param*> p_delete;
			for(int j =0; j < ps.size(); j++) {

				double dist = GetDistanceNorm(candidate_scaled, ps[j]->param_normalized);
				if(dist < mRadiusNeighbor) {
					n_compare += 1;

					if(ps[j]->reward < std::get<2>(candidate)) {
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
				to_be_deleted.push_back(std::pair<Eigen::VectorXd, std::vector<Param*>>(checklist[i], p_delete));
			}
		}
	}

	if(n_compare == 0) {
		mRecordLog.push_back("new parameter: " + vectorXd_to_string(nearest) + ", " + std::to_string(std::get<2>(candidate)));
	}
	if(flag) {

		for(int i = 0; i < to_be_deleted.size(); i++) {
			for(int j = 0; j < to_be_deleted[i].second.size(); j++) {
				Param* p = (to_be_deleted[i].second)[j];
				if(p->update) {
				//	mParamNew.erase(p->param_normalized);
				}
			}
			DeleteMappings(to_be_deleted[i].first, to_be_deleted[i].second);
		}

		Param* p = new Param();
		p->param_normalized = candidate_scaled;
		p->reward = std::get<2>(candidate);
		p->cps = std::get<0>(candidate);
		p->update = true;

	 	AddMapping(nearest, p);
	//	mParamNew.insert(std::pair<Eigen::VectorXd, Param*>(p->param_normalized, p));
	}
	return flag;

}
void
RegressionMemory::
SelectNewParamGoalCandidate() {
	mGoalCandidate.clear();
	while(mGoalCandidate.size() < mNumGoalCandidate) {
		double r = mUniform(mMT);
		r = std::floor(r * mGridMap.size());
		if(r == mGridMap.size())
			r -= 1;
		auto it_grid = std::next(mGridMap.begin(), (int)r);
		std::vector<Param*> params = it_grid->second->GetParams(); 
		if(params.size() == 0)
			continue;

		r = mUniform(mMT);
		r = std::floor(r * params.size());
		if(r == params.size())
			r -= 1;

		Eigen::VectorXd p = params[r]->param_normalized;
		Eigen::VectorXd dir(mDim);

		// for(int i = 0; i < mDim; i++) {
		// 	dir(i) =  mUniform(mMT) - 0.5;
		// }
		dir = p - mParamBVH->param_normalized;
		dir.normalize();

		for(int i = 0; i < mDim; i++) {
			p(i) += dir(i) * 1.5 * mParamGridUnit(i);
			if(p(i) > 1 || p(i) < 0) {
				p(i) = std::min(1.0, std::max(0.0, p(i)));
			} 
		}

		double d = GetDensity(p);
		if(d < 0.1) {
			mGoalCandidate.push_back(Denormalize(p));
		}
	}
}
void
RegressionMemory::
ResetExploration() {
	mNumActivatedPrev = mParamActivated.size();
	mTimeFromLastUpdate = 0;
	mPrevReward = 0;
	mPrevElite.clear();
	mPrevCPS.clear();
	mExplorationStep = 0;
	mIdxCandidate = -1;

	mGoalProgress.clear();
	for(int i = 0; i < mNumGoalCandidate; i++) {
		mGoalProgress.push_back(0);
	}
	mGoalExplored.clear();
	for(int i = 0; i < mNumGoalCandidate; i++) {
		mGoalExplored.push_back(false);
	}
	mGoalReward.clear();
	if(mGoalCandidate.size() != mNumGoalCandidate)
		return;
	for(int i = 0; i < mNumGoalCandidate; i++) {
		std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(mGoalCandidate[i]), mNumElite * 5);
		std::vector<std::pair<double, int>> ps_preward;
		for(int j = 0; j < ps.size(); j++) {
			double preward = GetParamReward(Denormalize(ps[j].second->param_normalized), mGoalCandidate[i]);
			ps_preward.push_back(std::pair<double, int>(preward, j));
		}

		std::stable_sort(ps_preward.begin(), ps_preward.end(), cmp_pair_int);

		double r = 0;
		for(int j = 0; j < mNumElite; j++) {
			r += ps_preward[j].first;
		}

		double currentReward = r / mNumElite;
		mGoalReward.push_back(currentReward);
	}

}
void
RegressionMemory::
EvalExplorationStep() {
	if(mGoalCandidate.size() != mNumGoalCandidate)
		return;
	mExplorationStep += 1;
	std::cout << "current progress: ";
	for(int i = 0; i < mNumGoalCandidate; i++) {
		if(mGoalExplored[i]) {
			std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(mGoalCandidate[i]), mNumElite * 5);
			std::vector<std::pair<double, int>> ps_preward;
			for(int j = 0; j < ps.size(); j++) {
				double preward = GetParamReward(Denormalize(ps[j].second->param_normalized), mGoalCandidate[i]);
				ps_preward.push_back(std::pair<double, int>(preward, j));
			}
			std::stable_sort(ps_preward.begin(), ps_preward.end(), cmp_pair_int);

			double r = 0;
			for(int j = 0; j < mNumElite; j++) {
				r += ps_preward[j].first;
			}

			double currentReward = r / mNumElite;
			mGoalProgress[i] = currentReward - mGoalReward[i];
			std::cout << "(" << currentReward << ", " <<  mGoalReward[i] << ", " <<  mGoalProgress[i] << ") ";

			mGoalReward[i] = currentReward;
			mGoalExplored[i] = false;
		}
	}
	std::cout << std::endl;
	mIdxCandidate = -1;
}
bool
RegressionMemory::
SetNextCandidate() {
	//mIdxCandidate = 0;

	if(mGoalCandidate.size() != mNumGoalCandidate)
		return false;
	int prevIdx = mIdxCandidate;
	if(mExplorationStep < 2) {
		mIdxCandidate += 1;
		if(mIdxCandidate >= mNumGoalCandidate)
			mIdxCandidate = 0;
	} else {
		if(mIdxCandidate == -1) {
			mIdxCandidate = 0;
		}
		for(int i = 0; i < mNumGoalCandidate; i++) {
			if(mGoalProgress[i] > mGoalProgress[mIdxCandidate])
				mIdxCandidate = i;
		}
		if(mGoalProgress[mIdxCandidate] <= 0) {
			double r = mUniform(mMT) * mNumGoalCandidate;
			mIdxCandidate = (int) std::floor(r);
			if(mIdxCandidate >= mNumGoalCandidate)
				mIdxCandidate -= 1;
		}
	}
	mGoalExplored[mIdxCandidate] = true;
	mParamGoalCur = mGoalCandidate[mIdxCandidate];

	return true;
}
bool 
RegressionMemory::
IsSpaceExpanded() { 
	std::cout << "ac: " << mParamActivated.size() <<", prev ac:" << mNumActivatedPrev << std::endl;
	int size = mParamActivated.size();
	if((size - mNumActivatedPrev) > mThresholdUpdate && size > 4 * mDim) {
		std::cout << "space expanded by " << mParamActivated.size() - mNumActivatedPrev << std::endl;
		mNumActivatedPrev = mParamActivated.size();
		mTimeFromLastUpdate = 0;
		return true;
	} 
	mTimeFromLastUpdate += 1;
	return false;
}
bool
RegressionMemory::
IsSpaceFullyExplored() {
	std::cout << "deac: " << mParamDeactivated.size() << ", ac: " << mParamActivated.size() << std::endl;
	if(mParamDeactivated.size() <= mThresholdUpdate) {
		return true;
	}
	return false;
}
void
RegressionMemory::
SaveContinuousParamSpace(std::string path) {
	Eigen::VectorXd base(mDim);
	base.setZero();
	std::vector<Eigen::VectorXd> points;
	points.push_back(base);
	for(int i = 0; i < mDim; i++) {
		std::vector<Eigen::VectorXd> vecs;
		double j = 0.1;
		while(j < 1) {
			auto iter = points.begin();
			while(iter != points.end()) {
				Eigen::VectorXd p = *iter;
				p(i) = j;
				vecs.push_back(p);
				iter++;
			}
			j += 0.1;
		}
		for(int j = 0; j < vecs.size(); j++) {
			points.push_back(vecs[j]);
		}	
	}
	std::ofstream ofs(path);

	for(int i = 0; i < points.size(); i++) {
		ofs << points[i].transpose() << " " << GetDensity(points[i]) << std::endl;
	}
	ofs.close();
}
double 
RegressionMemory::
GetParamReward(Eigen::VectorXd p, Eigen::VectorXd p_goal) {
	Eigen::VectorXd headRoot(6);
	headRoot << -1.77697, -1.73886, 0.793543, 0.00431308, 0.820601, -0.000182682;

	Eigen::Vector3d root_new = headRoot.segment<3>(0);
	root_new = projectToXZ(root_new);
	Eigen::AngleAxisd aa(root_new.norm(), root_new.normalized());
	Eigen::Vector3d dir = Eigen::Vector3d(p(0), 0, - sqrt(1 - p(0)*p(0)));
	dir *= p(2);
	Eigen::Vector3d p_hand = aa * dir;
	p_hand(1) = p(1);

	dir = Eigen::Vector3d(p_goal(0), 0, - sqrt(1 - p_goal(0)*p_goal(0)));
	dir *= p_goal(2);
	Eigen::Vector3d goal_hand = aa * dir;
	goal_hand(1) = p_goal(1);

	Eigen::Vector3d hand_diff = goal_hand - p_hand;
	double v_diff = p_goal(3) - p(3);
	
	double 	r_param = exp_of_squared(hand_diff,0.1) * exp(-pow(v_diff, 2)*150);

	return r_param;
}
void 
RegressionMemory::
SetParamGoal(Eigen::VectorXd paramGoal) { 
	mParamGoalCur = paramGoal; 
	auto pairs = GetNearestParams(Normalize(mParamGoalCur), 10);
	mRecordLog.push_back("new goal: " + vectorXd_to_string(mParamGoalCur));

	std::string result ="distance : ";
	for(int i = 0; i < pairs.size(); i++) {
		result += std::to_string(pairs[i].first) + " ";
	}
	mRecordLog.push_back(result);
}
void
RegressionMemory::
SetGoalInfo(double v) {
	mGoalInfo.param = mParamGoalCur;
	mGoalInfo.density = GetDensity(Normalize(mParamGoalCur));
	mGoalInfo.value = v;
	mGoalInfo.numSamples = mNumSamples;

	std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(mParamGoalCur), -1);
	
	std::vector<std::pair<double, int>> ps_preward;
	for(int i = 0; i < ps.size(); i++) {
		double preward = GetParamReward(Denormalize(ps[i].second->param_normalized), mParamGoalCur);
		ps_preward.push_back(std::pair<double, int>(preward, i));
	}

	std::stable_sort(ps_preward.begin(), ps_preward.end(), cmp_pair_int);
	std::vector<std::pair<double, Param*>> ps_elite;

	double r = 0;
	for(int i = 0; i < mNumElite; i++) {
		r += ps_preward[i].first;
	}
	mGoalInfo.rewards = r / mNumElite;

}
std::vector<Eigen::VectorXd> 
RegressionMemory::
GetCPSFromNearestParams(Eigen::VectorXd p_goal) {
	if(mGridMap.size() == 0) {
		return mParamBVH->cps;
	}

	// naive implementation
	std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(p_goal), mNumElite * 5);
	double r_baseline = GetParamReward(Denormalize(mParamBVH->param_normalized), p_goal);
	
	std::vector<std::pair<double, int>> ps_preward;
	for(int i = 0; i < ps.size(); i++) {
		double preward = GetParamReward(Denormalize(ps[i].second->param_normalized), p_goal);
		ps_preward.push_back(std::pair<double, int>(preward, i));
	}

	std::stable_sort(ps_preward.begin(), ps_preward.end(), cmp_pair_int);
	std::vector<std::pair<double, Param*>> ps_elite;
	double r = 0;
	for(int i = 0; i < mNumElite; i++) {
		r += ps_preward[i].first;
		int idx = ps_preward[i].second;

		if(r_baseline < ps_preward[i].first) {
			ps_elite.push_back(std::pair<double, Param*>(ps[idx].second->reward, ps[idx].second));
		} else {
			ps_elite.push_back(std::pair<double, Param*>(mParamBVH->reward, mParamBVH));
		}
	}
	double currentReward = r / mNumElite;
	// std::cout << "current reward: " << currentReward << std::endl;
	std::stable_sort(ps_elite.begin(), ps_elite.end(), cmp_pair_param);
	
	// if(mPrevReward >= currentReward) {
	// 	return mPrevCPS;
	// }
	mPrevReward = currentReward;
	// std::cout << "Elite Set Updated" <<std::endl;

	std::vector<Eigen::VectorXd> mean_cps;   
   	mean_cps.clear();
   	for(int i = 0; i < mNumKnots; i++) {
		mean_cps.push_back(Eigen::VectorXd::Zero(mDimDOF));
	}
   
	double weight_sum = 0;
	double weight_min = 1e8;
	std::vector<double> w;
	for(int i = 0; i < mNumElite; i++) {
		int idx = ps_preward[i].second;
		w.push_back(pow(ps_preward[i].first, 4)* ps[idx].second->reward);
		if(w.back() < weight_min) {
			weight_min = w.back();
		}
	}
	for(int i = 0; i < mNumElite; i++) {
		double w_i = w[i] - weight_min;
		weight_sum += w_i;
		int idx = ps_preward[i].second;
	  	std::vector<Eigen::VectorXd> cps = ps[idx].second->cps;
	    for(int j = 0; j < mNumKnots; j++) {
			mean_cps[j] += w_i * cps[j];
	    }
	}
	for(int i = 0; i < mNumKnots; i++) {
	    mean_cps[i] /= weight_sum;
	}
	if(mPrevElite.size() == 0) {
		for(int i = 0; i < mNumElite; i++) {
			mPrevElite.push_back(ps_elite[i]);
		}
		for(int i = 0; i < mNumKnots; i++) {
		    mPrevCPS.push_back(mean_cps[i]);
		}
	} else {
		for(int i = 0; i < mNumElite; i++) {
			mPrevElite[i] = ps_elite[i];
		}
		for(int i = 0; i < mNumKnots; i++) {
		    mPrevCPS[i] = mean_cps[i];
		}
	}

	return mean_cps;
}
void 
RegressionMemory::
SaveLog(std::string path) {
	std::ofstream ofs;
	ofs.open(path, std::fstream::out | std::fstream::app);
	ofs << std::to_string(mNumSamples) << std::endl;
	for(int i = 0; i < mRecordLog.size(); i++) {
		ofs << mRecordLog[i] << std::endl;
	}
	ofs.close();
	mRecordLog.clear();
}
void 
RegressionMemory::
SaveGoalInfo(std::string path) {

	if(mGoalInfo.rewards == -1)
		return;

	std::ofstream ofs;
	ofs.open(path, std::fstream::out | std::fstream::app);
	ofs << mGoalInfo.param.transpose() << std::endl;
	ofs << mGoalInfo.density << " " << mGoalInfo.value << " " << mGoalInfo.rewards << std::endl;


	std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(mParamGoalCur), -1);
	
	std::vector<std::pair<double, int>> ps_preward;
	for(int i = 0; i < ps.size(); i++) {
		double preward = GetParamReward(Denormalize(ps[i].second->param_normalized), mParamGoalCur);
		ps_preward.push_back(std::pair<double, int>(preward, i));
	}

	std::stable_sort(ps_preward.begin(), ps_preward.end(), cmp_pair_int);
	std::vector<std::pair<double, Param*>> ps_elite;

	double r = 0;
	for(int i = 0; i < mNumElite; i++) {
		r += ps_preward[i].first;
	}
	ofs << r / mNumElite - mGoalInfo.rewards << " " << mNumSamples - mGoalInfo.numSamples << std::endl;
	ofs.close();
}
};