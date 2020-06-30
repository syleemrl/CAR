#include "MultilevelSpline.h"
#include <Eigen/Dense>
namespace DPhy
{
Spline::
Spline(std::vector<double> knots, double end) {
	mEnd = end;
	mKnots = knots;
}
Spline::
Spline(double knot_interval, double end) {
	mEnd = end;
	mKnots.clear();
	for(int i = 0; i < end; i+= knot_interval) {
		mKnots.push_back(i);
	}
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
Approximate(std::vector<std::pair<Eigen::VectorXd,double>> motion) {
	mControlPoints.clear();
	mKnotMap.clear();
	int count = 0;
	for(int i = 0; i < motion.size(); i++) {
		if(count == mKnots.size())
			break;
		if(motion[i].second >= mKnots[count]) {
			mKnotMap.push_back(i);
			count += 1;
		}
	}
	
	int length = mKnots.size();
	int dof = motion[0].first.rows();

	Eigen::MatrixXd M(length, motion.size());
	Eigen::MatrixXd P(motion.size(), dof);
	Eigen::MatrixXd C(length, dof);
	M.setZero();
	P.setZero();
	count = 0;
	for(int i = 0; i < motion.size(); i++) {
		P.block(i, 0, 1, dof) = motion[i].first.transpose();
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
		C.block(0, i, length, 1) = cp;
	}

	mControlPoints.clear();
	for(int i = 0; i < C.rows(); i++) {
		mControlPoints.push_back(C.row(i));
	}

}
Eigen::VectorXd 
Spline::
GetPosition(double t) {
	int length = mKnots.size();
	Eigen::VectorXd p(mControlPoints[0].rows());
	p.setZero();
	int knot = 0;

	for(int i = length - 1; i >= 0; i--) {
		if(t > mKnots[i]) {
			knot = i;
			break;
		}
	}
	double knot_interval;
	if(knot + 1 >= mKnots.size()) {
		knot_interval = mKnots[0] + mEnd - mKnots[knot];
	} else {
		knot_interval = mKnots[knot + 1] - mKnots[knot];
	}
	t = (t - mKnots[knot]) / knot_interval;
	if(t < 0) {
		t += mEnd / knot_interval; 
	}
	for(int i = -1; i < 3; i++) {
		int cp_idx = (knot + i + length) % length; 
		p += this->B(i+1, t) * mControlPoints[cp_idx];
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
		std::vector<Eigen::VectorXd> cp = mSplines[i]->GetControlPoints();
		for(int j = 0; j < mEnd; j++) {
			motion[j] += mSplines[i]->GetPosition(j);
		}
	}

	return motion;
}
}
