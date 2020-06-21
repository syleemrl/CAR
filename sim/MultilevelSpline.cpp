#include "MultilevelSpline.h"

Spline::
Spline(int knot) {
	mKnot = knot;
}
double 
Spline::
B(int idx, int t) {
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
Approximate(std::vector<Eigen::VectorXd> pos) {
	mControlPoints.clear();
	mPositions.clear();

	int length = (int) std::ceil(pos.size() / mKnot) + 1;
	for(int i = -1; i < length + 1; i++) {
		Eigen::VectorXd cp(pos[0].rows());
		cp.setZero();
		double w_sqaure_sum = 0;
		for(int j = (i - 2) * mKnot; j < (i + 2) * mKnot; j++) {
			if(j < 0)
				continue;
			if(j >= pos.size())
				break;
			int floor_j = j / mKnot;
			double w = this->B(i + 1 - floor_j, (double)j / mKnot - floor_j);

			double bs = 0;
			for(int k = 0; k <= 3; k++) {
				bs += this->B(k, (double)j / mKnot - floor_j);
			}			
			Eigen::VectorXd beta(pos[0].rows());
			beta = w * pos[j] / bs;

			w_sqaure_sum += w * w;
			cp += w * w * beta;
		}	
		mControlPoints.push_back(cp);
	}

	for(int i = 0; i < pos.size(); i++) {
		Eigen::VectorXd p(pos[0].rows());
		double floor_t = i / mKnot;
		double t = (double) t / mKnot - floor_t;
		if(floor_t == length) {
			t = 1;
			floor_t -= 1;
		}
		p =   this->B(0, t) * mControlPoints[floor_t]
			+ this->B(1, t) * mControlPoints[floor_t + 1]
			+ this->B(2, t) * mControlPoints[floor_t + 2]
			+ this->B(3, t) * mControlPoints[floor_t + 3];
		mPositions.push_back(p);
	}
}
Eigen::VectorXd 
Spline::
GetPosition(double t) {
	if(t == std::floor(t))
		return mPositions[(int)t];
	else {
		Eigen::VectorXd p(pos[0].rows());
		double floor_t = t / mKnot;
		t = (double) t / mKnot - floor_t;
		p =   this->B(0, t) * mControlPoints[floor_t]
			+ this->B(1, t) * mControlPoints[floor_t + 1]
			+ this->B(2, t) * mControlPoints[floor_t + 2]
			+ this->B(3, t) * mControlPoints[floor_t + 3];
		return t;
	}
}
void
MultilevelSpline::
ConvertMotionToSpline(std::vector<Motion*> motions, int numLevels) {
}
std::vector<Motion*> 
MultilevelSpline::
ConvertSplineToMotion() {
	
}
