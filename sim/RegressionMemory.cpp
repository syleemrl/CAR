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
}
void
RegressionMemory::
InitParamSpace(Eigen::VectorXd paramBvh, std::pair<Eigen::VectorXd, Eigen::VectorXd> paramSpace, Eigen::VectorXd paramUnit, 
	double nDOF, double nknots) {
	mRecordLog.clear();

	mNumSamples = 1;
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

	mNumElite = 5;
	mRadiusNeighbor = 0.05;
	mThresholdInside = 0.6;
	mRangeExplore = 0.3;
	mThresholdActivate = 3;

	for(int i = 0; i < 2; i++) {
		mParamBVH = new Param();
		mParamBVH->cps.clear();
		for(int i = 0; i < mNumKnots; i++) {
			Eigen::VectorXd cps(mDimDOF);
			cps.setZero();
			mParamBVH->cps.push_back(cps);
		}

		mParamBVH->param_normalized = Normalize(paramBvh);
		mParamBVH->reward = 1;
		mParamBVH->update = 0;
		AddMapping(mParamBVH);
	}


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
double 
RegressionMemory::
GetVisitedRatio() {
	Eigen::VectorXd base(mDim);
	base = mParamGridUnit * 0.5;
	std::vector<Eigen::VectorXd> vecs_to_check;

	vecs_to_check.push_back(base);
	for(int i = 0; i < mDim; i++) {
		std::vector<Eigen::VectorXd> vecs;
	
		double range = std::floor(1.0 / mParamGridUnit(i) + 1e-8);
		double j = 1;
		while(j < range) {
			for(int k = 0; k < vecs_to_check.size(); k++) {
				Eigen::VectorXd p = vecs_to_check[k];
				p(i) = j * mParamGridUnit(i);
				vecs.push_back(p);
			}
			j += 0.5;
		}
		for(int j = 0; j < vecs.size(); j++) {
			vecs_to_check.push_back(vecs[j]);
		}	
	}

	double tot = vecs_to_check.size();
	double result = 0;
	for(int i = 0; i < vecs_to_check.size(); i++) {
		if(GetDensity(vecs_to_check[i]) > 0.3)
			result += 1;
	}
	result /= tot;

	return result;
}
void
RegressionMemory::
UpdateParamState() {
}
std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<double>>
RegressionMemory::
GetTrainingData() {
	mNumSamples = 0;
	std::vector<Eigen::VectorXd> x;
	std::vector<Eigen::VectorXd> y;
	std::vector<double> r;
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
			mNumSamples += 1;
		} 
		iter++;
	}
	std::cout << "num new data: " << r.size() << std::endl;

	mRecordLog.push_back("save training data: " + std::to_string(r.size()));
	return std::tuple<std::vector<Eigen::VectorXd>, 
					  std::vector<Eigen::VectorXd>, 
					  std::vector<double>> (x, y, r);
}
int
RegressionMemory::
GetNumSamples() {
	int n = 0;
	auto iter = mGridMap.begin();
	while(iter != mGridMap.end()) {
		std::vector<Param*> p = iter->second->GetParams();
		for(int i = 0; i < p.size(); i++) {
			n += 1;
		} 
		iter++;
	}
	return n;
}
void
RegressionMemory::
SaveParamSpace(std::string path) {
	auto x_y_r = GetTrainingData();

	std::vector<Eigen::VectorXd> x = std::get<0>(x_y_r);
	std::vector<Eigen::VectorXd> y = std::get<1>(x_y_r);
	std::vector<double> r = std::get<2>(x_y_r);

	std::ofstream ofs(path);
	
	ofs << r.size() << std::endl;
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
	mRecordLog.clear();
	mloadAllSamples= std::vector<Param*>();
	char buffer[256];

	std::ifstream is;
	is.open(path);

	if(is.fail())
		return;
	mGridMap.clear();

	is >> buffer;
	mNumSamples = atoi(buffer);

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
		p->update = 0;
		AddMapping(p);
		mloadAllSamples.push_back(p);

	}

	is.close();

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
		if((p_n(i) + radius * mParamGridUnit(i) > 0.5 * mParamGridUnit(i)) &&
			(p_n(i) - radius * mParamGridUnit(i) < -0.5 * mParamGridUnit(i))) {
			range(i) = 2;
		} else if(p_n(i) + radius * mParamGridUnit(i) > 0.5 * mParamGridUnit(i)) {
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
					n(i) += -2;
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
GetNearestParams(Eigen::VectorXd p, int n, bool search_neighbor, bool old, bool inside) {
	std::vector<std::pair<double, Param*>> params;
	if(search_neighbor) {
		std::vector<Eigen::VectorXd> grids = GetNeighborPointsOnGrid(p, 1);
		for(int i = 0; i < grids.size(); i++) {
			auto iter = mGridMap.find(grids[i]);
			if (iter != mGridMap.end()) {
				std::vector<Param*> ps = iter->second->GetParams();
				for(int j = 0; j < ps.size(); j++) {
					if(old) {
						if(!ps[j]->update && !inside)
							params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
						else if(!ps[j]->update && inside && GetDensity(ps[j]->param_normalized) >= mThresholdInside)
							params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
					} else {
						if(!inside)
							params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));			
						else if(inside && GetDensity(ps[j]->param_normalized) >= mThresholdInside)
							params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));			
					}
				}
			}
		}
	} else {
		auto iter = mGridMap.begin();
		while(iter != mGridMap.end()) {
			std::vector<Param*> ps = iter->second->GetParams();
			for(int j = 0; j < ps.size(); j++) {
				if(old) {
					if(!ps[j]->update && !inside)
						params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
					else if(!ps[j]->update && inside && GetDensity(ps[j]->param_normalized) >= mThresholdInside)
						params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));

				} else {
					if(!inside)
						params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));			
					else if(inside && GetDensity(ps[j]->param_normalized) >= mThresholdInside)
						params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));			
				}
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
	// std::cout << "delete" << std::endl;
	auto iter = mGridMap.find(nearest);
	// std::cout << nearest << std::endl;

	if (iter != mGridMap.end()) {
		ParamCube* pcube = iter->second;

		std::vector<Param*> ps_new;
		std::vector<Param*> ps_old = pcube->GetParams();
		std::vector<std::pair<Eigen::VectorXd, double>>	v;

		auto iter_trash = mTrashMap.find(nearest);
		if(iter_trash != mTrashMap.end())
			v = iter_trash->second;

		for(int i = 0; i < ps.size(); i++) {
			v.push_back(std::pair<Eigen::VectorXd, double>(ps[i]->param_normalized, ps[i]->reward));
		}
		while(v.size() > 100) {
			v.erase(v.begin());
		}

		if(iter_trash != mTrashMap.end())
			iter_trash->second = v;
		else
			mTrashMap.insert(std::pair<Eigen::VectorXd, std::vector<std::pair<Eigen::VectorXd, double>>>(nearest, v));



		int count = 0;
		for(int i = 0; i < ps_old.size(); i++) {
			if(count < ps.size() && IsEqualParam(ps_old[i], ps[count])) {
				delete ps_old[i];
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
	//std::cout << "delete done" << std::endl;

}
double 
RegressionMemory::
GetDensity(Eigen::VectorXd p, bool old) {
	double density = 0;
	double density_gaussian = 0;
	std::vector<Eigen::VectorXd> neighborlist = GetNeighborPointsOnGrid(p, 1);

	for(int j = 0; j < neighborlist.size(); j++) {
		auto iter = mGridMap.find(neighborlist[j]);
		if(iter != mGridMap.end()) {
			std::vector<Param*> ps = iter->second->GetParams();
			for(int k = 0; k < ps.size(); k++) {
				if(old && ps[k]->update) 
					continue;
				double d = GetDistanceNorm(p, ps[k]->param_normalized);

				density += 0.1 * std::max(0.0, 1 - d);
				density_gaussian += 0.1 * exp( - pow(d, 2) * 5);	
			}
		}
	}
	return density_gaussian;

}
std::pair<Eigen::VectorXd , bool>
RegressionMemory::
UniformSample(int visited) {
	int count = 0;
	while(1) {
		Eigen::VectorXd p(mDim);
		for(int i = 0; i < mDim; i++) {
			p(i) = mUniform(mMT);
		}
		
		// r = std::floor(r * mGridMap.size());
		// if(r == mGridMap.size())
		// 	r -= 1;
		// auto it_grid = std::next(mGridMap.begin(), (int)r);
		// std::vector<Param*> params = it_grid->second->GetParams(); 
		// if(params.size() == 0)
		// 	continue;

		// r = mUniform(mMT);
		// r = std::floor(r * params.size());
		// if(r == params.size())
		// 	r -= 1;
		// if(params[r]->update)
		// 	continue;
		// Eigen::VectorXd p = params[r]->param_normalized;
		// Eigen::VectorXd dir(mDim);

		// for(int i = 0; i < mDim; i++) {
		// 	dir(i) =  mUniform(mMT) - 0.5;
		// }
		// dir.normalize();

		// for(int i = 0; i < mDim; i++) {
		// 	r = mUniform(mMT);
		// 	p(i) += dir(i) * r * mParamGridUnit(i);
		// 	if(p(i) > 1 || p(i) < 0) {
		// 		p(i) = std::min(1.0, std::max(0.0, p(i)));
		// 	} 
		// }
		if(visited == -1) 
			return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);
		double d = GetDensity(p, true);

		if(!visited) {
			if(abs(p(0) - 1) < 1e-2 || abs(p(0)) < 1e-2) {
				continue;
			}
			if(mNumSamples < 10 && d > 0.05 && d < mThresholdInside) {
				return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);
			} else if (d < mThresholdInside && d > mThresholdInside - mRangeExplore) {
				return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);
			}
		}
		if(visited && d > mThresholdInside) {
			return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);
		}
		count += 1;
		if(!visited && count > 10000) {
			return std::pair<Eigen::VectorXd, bool>(Denormalize(p), false);
		}
	}
}
std::pair<Eigen::VectorXd , bool>
RegressionMemory::
UniformSample(double d0, double d1) {
	int count = 0;
	while(1) {
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
		if(params[r]->update)
			continue;
		Eigen::VectorXd p = params[r]->param_normalized;
		Eigen::VectorXd dir(mDim);

		for(int i = 0; i < mDim; i++) {
			dir(i) =  mUniform(mMT) - 0.5;
		}
		dir.normalize();

		for(int i = 0; i < mDim; i++) {
			r = mUniform(mMT);
			p(i) += dir(i) * r * mParamGridUnit(i);
			if(p(i) > 1 || p(i) < 0) {
				p(i) = std::min(1.0, std::max(0.0, p(i)));
			} 
		}
		double d = GetDensity(p, true);
		if(d >= d0 && d <= d1) {
			return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);
		}
		count += 1;
		if(count > 10000) {
			return std::pair<Eigen::VectorXd, bool>(Denormalize(p), false);
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

	Eigen::VectorXd candidate_scaled = Normalize(candidate_param);
	Eigen::VectorXd nearest = GetNearestPointOnGrid(candidate_scaled);

	std::vector<Eigen::VectorXd> checklist = GetNeighborPointsOnGrid(candidate_scaled, nearest, mRadiusNeighbor);
	int n_compare = 0;
	double prev_max = 0;
	bool flag = true;

	double update_max = 20;
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
					if(ps[j]->update > 0)
						ps[j]->update -= 1;
		
					if(prev_max < ps[j]->reward)
						prev_max = ps[j]->reward;
					if(ps[j]->reward < std::get<2>(candidate)) {
						p_delete.push_back(ps[j]);
						if(update_max < ps[j]->update || update_max == 20)
							update_max = ps[j]->update;
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
		// std::cout << "insert" << std::endl;

		double d = GetDensity(candidate_scaled);
		if(d > mThresholdInside) {
			for(int i = 0 ; i < checklist.size(); i++) {
				auto iter_trash = mTrashMap.find(checklist[i]);
				if (iter_trash != mTrashMap.end()) {
					std::vector<std::pair<Eigen::VectorXd, double>> flist = iter_trash->second;
					for(int j =0; j < flist.size(); j++) {
						double dist = GetDistanceNorm(candidate_scaled, flist[j].first);
						if(dist < mRadiusNeighbor) {
							if(flist[j].second < std::get<2>(candidate)) {
								update_max = 0;
							} else {
								flag = false;
								std::cout << "insert fail, cur: "  << std::get<2>(candidate) << " prev: " << flist[j].second  << std::endl;
								return flag;
							}
						} 
					}
				}
			}
		}
		// std::cout << 2 << std::endl;

		std::cout << candidate_scaled.transpose() << " " << std::get<2>(candidate) << " "<< update_max << std::endl;

		// std::cout << Denormalize(std::get<1>(candidate)).transpose() << " " <<to_be_deleted.size() << std::endl; 
		for(int i = 0; i < to_be_deleted.size(); i++) {
			for(int j = 0; j < to_be_deleted[i].second.size(); j++) {
				Param* p = (to_be_deleted[i].second)[j];
			}
			DeleteMappings(to_be_deleted[i].first, to_be_deleted[i].second);
		}
		// std::cout << "delete done" << std::endl;

		Param* p = new Param();
		p->param_normalized = candidate_scaled;
		p->reward = std::get<2>(candidate);
		p->cps = std::get<0>(candidate);
		p->update = std::max(0.0, update_max);

	 	AddMapping(nearest, p);
	
		if(GetDistanceNorm(candidate_scaled, Normalize(mParamGoalCur)) < 1.0 && to_be_deleted.size() == 0) {
			if(mUpdatedSamplesNearGoal == 0)
				mNewSamplesNearGoal = 1;
		} else if(GetDistanceNorm(candidate_scaled, Normalize(mParamGoalCur)) < 1.0 && p->reward >= prev_max + 0.01) {
			if(mNewSamplesNearGoal == 0)
				mUpdatedSamplesNearGoal = 1;
		}
		// std::cout << "insert done" << std::endl;

	}

	return flag;
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
	Eigen::VectorXd diff = p - p_goal;
	double r_param = exp_of_squared(diff, 0.2);
	return r_param;
}
void 
RegressionMemory::
SetParamGoal(Eigen::VectorXd paramGoal) { 
	mParamGoalCur = paramGoal; 
	auto pairs = GetNearestParams(Normalize(mParamGoalCur), 10, false, true);
	mRecordLog.push_back("new goal: " + vectorXd_to_string(mParamGoalCur));
	mEliteGoalDistance = 0;
	std::string result ="distance : ";
	for(int i = 0; i < pairs.size(); i++) {
		mEliteGoalDistance += pairs[i].first;
		result += std::to_string(pairs[i].first) + " ";
	}
	mEliteGoalDistance /= pairs.size();
	mNewSamplesNearGoal = 0;
	mUpdatedSamplesNearGoal = 0;

	mRecordLog.push_back(result);
}
std::vector<Eigen::VectorXd> 
RegressionMemory::
GetCPSFromNearestParams(Eigen::VectorXd p_goal) {
	// naive implementation
	std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(p_goal), mNumElite * 10, false, true);
	// std::cout << p_goal.transpose() << " " << GetDensity(Normalize(p_goal)) << std::endl;
	if(ps.size() < mNumElite) {
		return mParamBVH->cps;
	}

	double f_baseline = GetParamReward(Denormalize(mParamBVH->param_normalized), p_goal);
	std::vector<std::pair<double, Param*>> ps_elite;
	double r = 0;
	for(int i = 0; i < ps.size(); i++) {
		double preward = GetParamReward(Denormalize(ps[i].second->param_normalized), p_goal);
		double fitness = preward*ps[i].second->reward;
		// std::cout << Denormalize(ps[i].second->param_normalized).transpose() << " " << preward << " " << ps[i].second->reward << " / " <<fitness << std::endl;
	//	if(f_baseline < fitness) {
			ps_elite.push_back(std::pair<double, Param*>(fitness, ps[i].second));
		// } else {
		// 	ps_elite.push_back(std::pair<double, Param*>(f_baseline, mParamBVH));

		// }
		r += preward;

	}
	
	// double currentReward = r / mNumElite;
	// std::cout << "current reward: " << currentReward << std::endl;
	std::stable_sort(ps_elite.begin(), ps_elite.end(), cmp_pair_param);
	
	// if(mPrevReward >= currentReward) {
	// 	return mPrevCPS;
	// }
	// mPrevReward = currentReward;
	// std::cout << "Elite Set Updated" <<std::endl;
	std::vector<Eigen::VectorXd> mean_cps;   
   	mean_cps.clear();
   	for(int i = 0; i < mNumKnots; i++) {
		mean_cps.push_back(Eigen::VectorXd::Zero(mDimDOF));
	}
	double weight_sum = 0;
	for(int i = 0; i < mNumElite; i++) {
		double w = ps_elite[i].first;
		weight_sum += w;
	    std::vector<Eigen::VectorXd> cps = ps_elite[i].second->cps;
	    for(int j = 0; j < mNumKnots; j++) {
			mean_cps[j] += w * cps[j];
	    }
	}

	for(int i = 0; i < mNumKnots; i++) {
	    mean_cps[i] /= weight_sum;
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
double
RegressionMemory::
GetFitness(Eigen::VectorXd p) {
	std::vector<std::pair<double, Param*>> ps = GetNearestParams(p, mNumElite * 5, false, true);
	if(ps.size() < mNumElite) {
		return 1;
	}

	double f_baseline = GetParamReward(Denormalize(mParamBVH->param_normalized), Denormalize(p));
	std::vector<std::pair<double, Param*>> ps_elite;
	for(int i = 0; i < ps.size(); i++) {
		double preward = GetParamReward(Denormalize(ps[i].second->param_normalized), Denormalize(p));
		double fitness = preward*pow(ps[i].second->reward, 2);
		ps_elite.push_back(std::pair<double, Param*>(fitness, ps[i].second));
	}

	std::stable_sort(ps_elite.begin(), ps_elite.end(), cmp_pair_param);
	
	double fitness = 0;
	for(int i = 0; i < mNumElite; i++) {
		fitness += ps_elite[i].second->reward;
	}
	return fitness / mNumElite;

}
std::tuple<std::vector<Eigen::VectorXd>, 
   		   std::vector<Eigen::VectorXd>,  
		   std::vector<double>, 
		   std::vector<double>>
RegressionMemory::
GetParamSpaceSummary() {
	Eigen::VectorXd base = 0.05 * Eigen::VectorXd::Ones(mDim);
	std::vector<Eigen::VectorXd> grids;
	std::vector<Eigen::VectorXd> grids_denorm;
	std::vector<double> fitness;
	std::vector<double> density;

	grids.push_back(base);
	density.push_back(GetDensity(base));
	if(density.back() > 0.1) {
		fitness.push_back(GetFitness(base));
	} else {
		fitness.push_back(0);
	}

	for(int i = 0; i < mDim; i++) {
		std::vector<Eigen::VectorXd> vecs;
		double j = 0.1;
		while(j < 1.0) {
			auto iter = grids.begin();
			while(iter != grids.end()) {
				Eigen::VectorXd p = *iter;
				p(i) = j;
				vecs.push_back(p);
				iter++;
			}
			j += 0.05;
		}
		for(int j = 0; j < vecs.size(); j++) {
			grids.push_back(vecs[j]);
			density.push_back(GetDensity(vecs[j]));
			if(density.back() > 0.1) {
				fitness.push_back(GetFitness(vecs[j]));
			} else {
				fitness.push_back(0);
			}
		}	
	}
	for(int i = 0; i < grids.size(); i++) {
		grids_denorm.push_back(Denormalize(grids[i]));
	}
	return std::tuple<std::vector<Eigen::VectorXd>, 
		   std::vector<Eigen::VectorXd>, 
		   std::vector<double>, 
		   std::vector<double>>(grids_denorm, grids, fitness, density);
}
double 
RegressionMemory::
GetFitnessMean() {

	int count = 0;
	double fitness = 0;
	auto iter = mGridMap.begin();
	while(iter != mGridMap.end()) {
		std::vector<Param*> p = iter->second->GetParams();
		for(int i = 0; i < p.size(); i++) {
			fitness += p[i]->reward;
			count += 1;
		} 
		iter++;
	}
	if(count == 0)
		return 0;

	return fitness / count;
}
};