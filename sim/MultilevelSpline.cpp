#include "MultilevelSpline.h"
#include <Eigen/Dense>
namespace DPhy
{
Spline::
Spline(std::vector<double> knots, double end) {
	mEnd = end;
	mKnots = knots;
	mNC_idxs.clear();


}
Spline::
Spline(double knot_interval, double end) {
	mEnd = end;
	mKnots.clear();
	for(int i = 0; i < end; i+= knot_interval) {
		mKnots.push_back(i);
	}

	mNC_idxs.clear();

}
void
Spline:: 
SetKnots(std::vector<double> knots) {
	mKnots = knots;
}
void
Spline:: 
SetKnots(double knot_interval) {
	mKnots.clear();
	for(int i = 0; i < mEnd; i+= knot_interval) {
		mKnots.push_back(i);
	}

}
double 
Spline::
B(int idx, double t) {
	if(idx == 0) {
		return 1.0 / 6 * pow(1 - t, 3);
	} else if (idx == 1) {
		return 1.0 / 6 * (3 * pow(t, 3) - 6 * pow(t, 2) + 4);
	} else if (idx == 2) {
		return 1.0 / 6 * (-3 * pow(t, 3) + 3 * pow(t, 2) + 3 * t + 1);
	} else {
		return 1.0 / 6 * pow(t, 3);
	}
}
void
Spline::
Approximate(std::vector<std::pair<Eigen::VectorXd,double>> motion, std::vector<int> idxs, bool circular) {
	int length = mKnots.size();
	Eigen::MatrixXd P(motion.size(), idxs.size());
	Eigen::MatrixXd C(length+3, idxs.size());

	if(circular) {
		Eigen::MatrixXd M(length, motion.size());

		M.setZero();

		int count = 0;
		for(int i = 0; i < motion.size(); i++) {
			for(int j = 0; j < idxs.size(); j++) {
				P(i, j) = (motion[i].first)[idxs[j]];
			}
			if(count + 1 < mKnots.size() && motion[i].second >= mKnots[count + 1]) {
				count += 1;
			}
			double interval = mKnots[ (count + 1) % length ] - mKnots[count];
			if(interval < 0)
				interval += mEnd;
			double f = (motion[i].second - mKnots[count]) / interval;
			M((count - 1 + length) % length, i) = B(0, f);
			M(count, i) = B(1, f);
			M((count + 1) % length, i) = B(2, f);
			M((count + 2) % length, i) = B(3, f);
		}	
		
		Eigen::MatrixXd Mt = M.transpose();
		auto solver = Mt.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

		for(int i = 0; i < P.cols(); i++) {
			Eigen::VectorXd cp = solver.solve(P.col(i));
			C.block(1, i, length, 1) = cp;
			C(0, i) = cp(cp.rows()-1);
			C(length+1, i) = cp(0);
			C(length+2, i) = cp(1);
		}
	} else {
		Eigen::MatrixXd M(length+3, motion.size());

		M.setZero();

		int count = 0;
		for(int i = 0; i < motion.size(); i++) {
			for(int j = 0; j < idxs.size(); j++) {
				P(i, j) = (motion[i].first)[idxs[j]];
			}			
			if(count + 1 < mKnots.size() && motion[i].second >= mKnots[count + 1]) {
				count += 1;
			}

			double interval;
			if(count + 1 >= mKnots.size())
				interval = mEnd - mKnots[count];
			else
				interval = mKnots[count + 1] - mKnots[count];
			double f = (motion[i].second - mKnots[count]) / interval;

			M((count - 1) + 1, i) = B(0, f);
			M(count + 1, i) = B(1, f);
			M((count + 1) + 1, i) = B(2, f);
			
			if((count + 2) + 1 < M.rows())
				M((count + 2) + 1, i) = B(3, f);
		}	
		
		Eigen::MatrixXd Mt = M.transpose();
		auto solver = Mt.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

		for(int i = 0; i < P.cols(); i++) {
			Eigen::VectorXd cp = solver.solve(P.col(i));
			C.block(0, i, length + 3, 1) = cp;
		}
	}

	for(int i = 0; i < C.rows(); i++) {
		for(int j = 0; j < idxs.size(); j++) {
			mControlPoints[i][idxs[j]] = C(i, j);
		}
	}
}
void
Spline::
Approximate(std::vector<std::pair<Eigen::VectorXd,double>> motion) {
	mControlPoints.clear();

	int dof = motion[0].first.rows();

	for(int i = 0; i < mKnots.size()+3; i++) {
		Eigen::VectorXd cp(dof);
		cp.setZero();
		mControlPoints.push_back(cp);
	}
	int count = 0;
	std::vector<int> C_idxs;

	for(int i = 0; i < dof; i++) {
		if(mNC_idxs.size() == 0 || i != mNC_idxs[count]) {
			C_idxs.push_back(i);
		} else if(count + 1 < mNC_idxs.size()) {
			count += 1;
		}
	}

	this->Approximate(motion, C_idxs, true);
	this->Approximate(motion, mNC_idxs, false);

}

Eigen::VectorXd 
Spline::
GetPosition(double t) {

	int length = mKnots.size();
	int dof = mControlPoints[0].rows();
	Eigen::VectorXd p(dof);
	p.setZero();
	int knot = 0;

	for(int i = length - 1; i >= 0; i--) {
		if(t > mKnots[i]) {
			knot = i;
			break;
		}
	}
	int nc_count = 0;
	double knot_interval_c;
	double knot_interval_nc;
	if(knot + 1 >= mKnots.size()) {
		knot_interval_c = mKnots[0] + mEnd - mKnots[knot];
		knot_interval_nc = mEnd - mKnots[knot];
	} else {
		knot_interval_c = mKnots[knot + 1] - mKnots[knot];
		knot_interval_nc = knot_interval_c;
	}
	double t_c = (t - mKnots[knot]) / knot_interval_c;
	double t_nc = (t - mKnots[knot]) / knot_interval_nc;

	bool circular = true;
	for(int i = 0; i < dof; i++) {
		if(mNC_idxs.size() > nc_count && i == mNC_idxs[nc_count]) {
			circular = false;
			nc_count += 1;
		} 
		for(int j = -1; j < 3; j++) {
			int cp_idx = knot + j + 1; 
			if(circular)
				p(i) += this->B(j+1, t_c) * mControlPoints[cp_idx](i);
			else
				p(i) += this->B(j+1, t_nc) * mControlPoints[cp_idx](i);
		}
	}

	return p;

}
void
Spline::
Save(std::string path) {

	std::ofstream ofs(path);

	ofs << mKnots.size() << std::endl;
	for(auto t: mKnots) {
		ofs << t << std::endl;
	}
	for(auto t: mControlPoints) {
		ofs << t.transpose() << std::endl;
	}
	std::cout << "saved spline to " << path << std::endl;
	ofs.close();
}
MultilevelSpline::
MultilevelSpline(int level, double end) {
	mNumLevels = level;
	mEnd = end;	
	for(int i = 0; i < mNumLevels; i++) {
		Spline* s = new Spline(1, mEnd);
		mSplines.push_back(s);
	}
}
MultilevelSpline::
MultilevelSpline(int level, double end, std::vector<int> nc_idx) {
	mNumLevels = level;
	mEnd = end;
	int count = 0;
	for(int i = 0; i < mNumLevels; i++) {
		Spline* s = new Spline(1, mEnd);
		s->SetNonCircular(nc_idx);
		mSplines.push_back(s);
	}
}
MultilevelSpline::
~MultilevelSpline(){	
	while(!mSplines.empty()){
		Spline* s = mSplines.back();
		mSplines.pop_back();

		delete s;
	}	
}
void
MultilevelSpline::
SetKnots(int i, std::vector<double> knots) {
	mSplines[i]->SetKnots(knots);
}
void
MultilevelSpline::
SetKnots(int i, double knot_interval) {
	mSplines[i]->SetKnots(knot_interval);
}
void 
MultilevelSpline::
SetControlPoints(int i, std::vector<Eigen::VectorXd> cps) {
	mSplines[i]->SetControlPoints(cps);
}
std::vector<Eigen::VectorXd> 
MultilevelSpline::
GetControlPoints(int i) {
	return mSplines[i]->GetControlPoints();
}
void
MultilevelSpline::
ConvertMotionToSpline(std::vector<std::pair<Eigen::VectorXd,double>> motion) {
	for(int i = 0; i < mNumLevels; i++) {
		if(i == 0)
		{
			mSplines[i]->Approximate(motion);
		}
		else {
			std::vector<std::pair<Eigen::VectorXd,double>> displacement;	
			for(int j = 0; j < motion.size(); j++) {
				Eigen::VectorXd p = mSplines[i-1]->GetPosition(motion[j].second);
				displacement.push_back(std::pair<Eigen::VectorXd,double>(motion[j].first - p, motion[j].second));
			}
			mSplines[i]->Approximate(displacement);
		}
	}
}
std::vector<Eigen::VectorXd> 
MultilevelSpline::
ConvertSplineToMotion() {
	std::vector<Eigen::VectorXd> motion;
	
	int dof = mSplines[0]->GetPosition(0).rows();

	for(int i = 0; i < mEnd; i++) {
		motion.push_back(Eigen::VectorXd::Zero(dof));
	}

	for(int i = 0; i < mNumLevels; i++) {
		for(int j = 0; j < mEnd; j++) {
			motion[j] += mSplines[i]->GetPosition(j);
		}
	}

	return motion;
}
}