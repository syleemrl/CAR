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
	mRadiusNeighbor = 0.2;
	mThresholdActivate = 3;
	mThresholdUpdate = 2 * mDim;
	mNumGoalCandidate = 30;

	for(int j = 0; j < 1; j++) {
		mParamBVH = new Param();
		mParamBVH->cps.clear();
		for(int i = 0; i < mNumKnots; i++) {
			Eigen::VectorXd cps(mDimDOF);
			cps.setZero();
			mParamBVH->cps.push_back(cps);
		}

		mParamBVH->param_normalized = Normalize(paramBvh);
		mParamBVH->reward = 1;
		mParamBVH->update = false;
		AddMapping(mParamBVH);

	}
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
	auto iter = mParamNew.begin();
	while(iter != mParamNew.end()) {
		iter->second->update = false;
		iter++;
	}
	mParamNew.clear();
}

std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<double>>
RegressionMemory::
GetTrainingData(bool old) {
	if(old)
		mNumSamples = 0;
	std::vector<Eigen::VectorXd> x;
	std::vector<Eigen::VectorXd> y;
	std::vector<double> r;
	auto iter = mGridMap.begin();
	while(iter != mGridMap.end()) {
		std::vector<Param*> p = iter->second->GetParams();
		for(int i = 0; i < p.size(); i++) {
			if(old && p[i]->update ) {
				p[i]->update = false;
			}
			for(int j = 0; j < mNumKnots; j++) {
				Eigen::VectorXd x_elem(mDim + 1);
				x_elem << j, p[i]->param_normalized;
				x.push_back(x_elem);
				y.push_back((p[i]->cps)[j]);
			}
			r.push_back(p[i]->reward);
			if(old)
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
void
RegressionMemory::
SaveParamSpace(std::string path, bool old) {
	auto x_y_r = GetTrainingData(old);

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
	mNumSamples = 1;
	mRecordLog.clear();
	mloadAllSamples= std::vector<Param*>();

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
		mloadAllSamples.push_back(p);
		mNumSamples += 1;
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
GetNearestParams(Eigen::VectorXd p, int n, bool search_neighbor, bool old) {
	std::vector<std::pair<double, Param*>> params;
	if(search_neighbor) {
		std::vector<Eigen::VectorXd> grids = GetNeighborPointsOnGrid(p, 1);
		for(int i = 0; i < grids.size(); i++) {
			auto iter = mGridMap.find(grids[i]);
			if (iter != mGridMap.end()) {
				std::vector<Param*> ps = iter->second->GetParams();
				for(int j = 0; j < ps.size(); j++) {
					if(old) {
						if(!ps[j]->update)
							params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
					} else 
						params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
				}
			}
		}
	} else {
		auto iter = mGridMap.begin();
		while(iter != mGridMap.end()) {
			std::vector<Param*> ps = iter->second->GetParams();
			for(int j = 0; j < ps.size(); j++) {
				if(old) {
					if(!ps[j]->update)
						params.push_back(std::pair<double, Param*>(GetDistanceNorm(p, ps[j]->param_normalized), ps[j]));
				} else 
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
				// std::cout << Denormalize(ps_old[i]->param_normalized).transpose() << std::endl;

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
UniformSample(bool visited) {
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
		if(!visited) {
			// std::cout<<"mNumSamples : "<<mNumSamples<<std::endl;
			// std::cout<<"d : "<<d<<std::endl;
			// std::cout<<"p : "<<Denormalize(p).transpose()<<"("<<p.transpose()<<")"<<std::endl;
			if(mNumSamples <= 10 && d < 0.5 && d > 0.05)
				return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);

			if (d < 0.5 && d > 0.2) // use blended 
				return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);
		}
		if(visited && d > 0.6) { // increse d ? // use regression
			return std::pair<Eigen::VectorXd, bool>(Denormalize(p), true);
		}
		count += 1;
		if(!visited && count > 10000) {
			return std::pair<Eigen::VectorXd, bool>(p, false);
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
	Eigen::VectorXd candidate_scaled = (candidate_param - mParamMin).cwiseProduct(mParamScale); //normalize
	Eigen::VectorXd nearest = GetNearestPointOnGrid(candidate_scaled);
	//std::cout << candidate_param.transpose() << " / " << candidate_scaled.transpose() << std::endl;
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
		// std::cout << "Added: " <<  std::get<1>(candidate).transpose() << std::endl;

		for(int i = 0; i < to_be_deleted.size(); i++) {
			// for(int j = 0; j < to_be_deleted[i].second.size(); j++) {
			// 	Param* p = (to_be_deleted[i].second)[j];
			// 	if(p->update) {
			// 	//	mParamNew.erase(p->param_normalized);
			// 	}
			// }
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

		for(int i = 0; i < mDim; i++) {
			dir(i) =  mUniform(mMT) - 0.5;
		}
		// dir = p - mParamBVH->param_normalized;
		dir.normalize();

		for(int i = 0; i < mDim; i++) {
			p(i) += dir(i) * 1.0 * mParamGridUnit(i);
			if(p(i) > 1 || p(i) < 0) {
				p(i) = std::min(1.0, std::max(0.0, p(i)));
			} 
		}
		double d = GetDensity(p, true);

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
	mGoalUpdate.clear();
	mCPSCandidate.clear();
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
		int size = ps.size();
		size = std::min(mNumElite, size);		
		for(int j = 0; j < size; j++) {
			r += ps_preward[j].first;
		}

		double currentReward = r / size;
		mGoalReward.push_back(currentReward);
		mCPSCandidate.push_back(GetCPSFromNearestParams(mGoalCandidate[i]));
		mGoalUpdate.push_back(0);
	}

}
void
RegressionMemory::
EvalExplorationStep() {

	// if(mGoalCandidate.size() != mNumGoalCandidate)
	// 	return;
	// mExplorationStep += 1;
	// std::cout << "current progress: ";
	// for(int i = 0; i < mNumGoalCandidate; i++) {
	// 	if(mGoalExplored[i]) {
	// 		std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(mGoalCandidate[i]), mNumElite * 5);
	// 		std::vector<std::pair<double, int>> ps_preward;
	// 		for(int j = 0; j < ps.size(); j++) {
	// 			double preward = GetParamReward(Denormalize(ps[j].second->param_normalized), mGoalCandidate[i]);
	// 			ps_preward.push_back(std::pair<double, int>(preward, j));
	// 		}
	// 		std::stable_sort(ps_preward.begin(), ps_preward.end(), cmp_pair_int);

	// 		double r = 0;
	// 		int size = ps.size();
	// 		size = std::min(mNumElite, size);
	// 		for(int j = 0; j < size; j++) {
	// 			r += ps_preward[j].first;
	// 		}

	// 		double currentReward = r / size;
	// 		mGoalProgress[i] = currentReward - mGoalReward[i];
	// 		std::cout << "(" << currentReward << ", " <<  mGoalReward[i] << ", " <<  mGoalProgress[i] << ") ";

	// 		mGoalReward[i] = currentReward;
	// 		mGoalExplored[i] = false;
	// 	}
	// }
	// std::cout << std::endl;
	// mIdxCandidate = -1;
}
std::vector<Eigen::VectorXd>
RegressionMemory::
GetCurrentCPS() {
	if(mIdxCandidate == -1)
		return mCPSCandidate[0];
	return mCPSCandidate[mIdxCandidate];
}
bool
RegressionMemory::
SetNextCandidate() {

	if(mGoalCandidate.size() != mNumGoalCandidate)
		return false;
	int prevIdx = mIdxCandidate;
	//if(mExplorationStep < 5) {
		mIdxCandidate += 1;
		if(mIdxCandidate >= mNumGoalCandidate)
			mIdxCandidate = 0;
	// } else {
	// 	if(mIdxCandidate == -1) {
	// 		mIdxCandidate = 0;
	// 	}
	// 	for(int i = 0; i < mNumGoalCandidate; i++) {
	// 		if(mGoalProgress[i] > mGoalProgress[mIdxCandidate])
	// 			mIdxCandidate = i;
	// 	}
	// 	if(mGoalProgress[mIdxCandidate] <= 0) {
	// 		double r = mUniform(mMT) * mNumGoalCandidate;
	// 		mIdxCandidate = (int) std::floor(r);
	// 		if(mIdxCandidate >= mNumGoalCandidate)
	// 			mIdxCandidate -= 1;
	// 	}
	// }
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
	Eigen::VectorXd l_diff = p_goal - p;
	// l_diff *= 0.1;
	double r_param = exp_of_squared(l_diff, 0.1);

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
	// naive implementation
	std::vector<std::pair<double, Param*>> ps = GetNearestParams(Normalize(p_goal), mNumElite * 5, false, true);

	if(ps.size() < mNumElite) {
		return mParamBVH->cps;
	}

	double f_baseline = GetParamReward(Denormalize(mParamBVH->param_normalized), p_goal);
	
	// std::cout << "cps/ "<<p_goal.transpose()<<"/";
	std::vector<std::pair<double, Param*>> ps_elite;
	double r = 0;
	for(int i = 0; i < ps.size(); i++) {
		double preward = GetParamReward(Denormalize(ps[i].second->param_normalized), p_goal);
		double fitness = preward*pow(ps[i].second->reward, 2);
		// std::cout << p_goal.transpose()<<"/ "<<Denormalize(ps[i].second->param_normalized).transpose() << "/ " << preward << "/ " << ps[i].second->reward << "/ " << fitness << std::endl;
		 // if(ps[i].second->reward == 1)
		 // 	continue;
		//if(f_baseline < fitness) {
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
		double w = log(mNumElite + 1) - log(i + 1);
		weight_sum += w;
	    std::vector<Eigen::VectorXd> cps = ps_elite[i].second->cps;
		// std::cout<<"/ "<<Denormalize(ps_elite[i].second->param_normalized).transpose() << "/ " << ps_elite[i].second->reward << "/ " << ps_elite[i].first << std::endl;
	
	    for(int j = 0; j < mNumKnots; j++) {
			mean_cps[j] += w * cps[j];
	    }
	}

	for(int i = 0; i < mNumKnots; i++) {
	    mean_cps[i] /= weight_sum;
		// std::cout << i << " " << exp(mean_cps[i][mDimDOF - 1]) << std::endl;
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