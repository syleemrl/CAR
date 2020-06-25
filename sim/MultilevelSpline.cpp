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
	for(int i = 0; i <motion.size(); i++) {
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
	for(int i = 0; i < C.rows(); i++) {
		mControlPoints.push_back(C.row(i));
	}
	// int length = mKnots.size();
	// std::vector<double> weight;
	// std::vector<Eigen::VectorXd> cp;
	// for(int i = 0; i < length; i++) {
	// 	cp.push_back(Eigen::VectorXd::Zero(motion[0].first.rows()));
	// 	weight.push_back(0);
	// }
	// count = 0;
	// for(int i = 0; i < motion.size(); i++) {
	// 	if(count+1 < mKnots.size() && motion[i].second >= mKnotMap[count + 1]) {
	// 		count += 1;
	// 	}
	// 	double t = motion[i].second - mKnotMap[count];
	// 	double interval;
	// 	if(count + 1 < mKnots.size()) {
	// 		interval = mKnotMap[count+1] - mKnotMap[count];
	// 	} else {
	// 		interval = mEnd - mKnotMap[count];
	// 	}
	// 	double f = t / interval;
	// 	std::cout << i << " " << f << " " << count << std::endl;
	// 	double r0 = B(0, f);
	// 	double r1 = B(1, f);
	// 	double r2 = B(2, f);
	// 	double r3 = B(3, f);

	// 	double sum = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
	// 	cp[(count - 1 + length) % length] += r0 * r0 * motion[i].first * r0 / sum;
	// 	cp[count % length] += r1 * r1 * motion[i].first * r1 / sum;
	// 	cp[(count + 1) % length] += r2 * r2 * motion[i].first * r2 / sum;
	// 	cp[(count + 2) % length] += r3 * r3 * motion[i].first * r3 / sum;

	// 	weight[(count - 1 + length) % length] += r0 * r0;
	// 	weight[count % length] += r1 * r1;
	// 	weight[(count + 1) % length] += r2 * r2;
	// 	weight[(count + 2) % length] += r3 * r3;

	// }
	// for(int i = 0; i < length; i++) {
	// 	cp[i] /= weight[i];
	// 	mControlPoints.push_back(cp[i]);
	// 	std::cout << cp[i].segment<6>(0).transpose() << std::endl;
	// }


	// for(int i = 0; i < length; i++) {
	// 	std::cout << mKnots[i] << std::endl;
	// 	Eigen::VectorXd cp(motion[0].first.rows());
	// 	cp.setZero();
	// 	double w_square_sum = 0;

	// 	int b_idx = 3;
	// 	int neighbor_idx_knot = (i - 2 + length) % length;
	// 	int neighbor_idx_knot_next = (i - 1 + length) % length;
	// 	double knot_interval;
	// 	if(neighbor_idx_knot > neighbor_idx_knot_next) {
	// 		knot_interval = mKnots[neighbor_idx_knot_next] + mEnd - mKnots[neighbor_idx_knot];
	// 	} else {
	// 		knot_interval = mKnots[neighbor_idx_knot_next] - mKnots[neighbor_idx_knot];
	// 	}

	// 	int neighbor_idx_motion = mKnotMap[neighbor_idx_knot];
	// 	double weight_sum = 0;
	// 	while(1) {
	// 		std::cout << "neighbor: " << mKnots[neighbor_idx_knot] << " " << neighbor_idx_motion << std::endl;
	// 		double t = (motion[neighbor_idx_motion].second - mKnots[neighbor_idx_knot]) / knot_interval;
	// 		if(t < 0) {
	// 			t += mEnd / knot_interval; 
	// 		}
	// 		double w = this->B(b_idx, t);
	// 		double b_sum = pow(this->B(0, t), 2) + pow(this->B(1, t), 2) + pow(this->B(2, t), 2) + pow(this->B(3, t), 2);
	// 		Eigen::VectorXd beta = w * motion[neighbor_idx_motion].first / b_sum;
	// 		cp += w * w * beta;
	// 		weight_sum += w / b_sum;
	// 		w_square_sum += w * w;
			
	// 		neighbor_idx_motion += 1;
	// 		if(neighbor_idx_motion >= mEnd)
	// 			neighbor_idx_motion = 0;

	// 		if(mKnotMap[neighbor_idx_knot_next] == neighbor_idx_motion) {
	// 			neighbor_idx_knot = (neighbor_idx_knot + 1) % length;
	// 			neighbor_idx_knot_next = (neighbor_idx_knot + 1) % length;
	// 			b_idx -= 1;
				
	// 			if(neighbor_idx_knot > neighbor_idx_knot_next) {
	// 				knot_interval = mKnotMap[neighbor_idx_knot_next] + mEnd - mKnotMap[neighbor_idx_knot];
	// 			} else {
	// 				knot_interval = mKnotMap[neighbor_idx_knot_next] - mKnotMap[neighbor_idx_knot];
	// 			}
	// 		}
			
	// 		if(((neighbor_idx_knot - i + length) % length) == 2)
	// 			break;
	// 	}

	// 	cp /= w_square_sum;
	// 	mControlPoints.push_back(cp);
	// }

}
Eigen::VectorXd 
Spline::
GetPosition(double t) {
	int length = mKnots.size();

	Eigen::VectorXd p(mControlPoints[0].rows());
	p.setZero();
	int knot = 0;

	for(int i = length - 1; i >= 0; i--) {
		if(t > mKnotMap[i]) {
			knot = i;
			break;
		}
	}
	double knot_interval;
	if(knot + 1 >= mKnots.size()) {
		knot_interval = mKnotMap[0] + mEnd - mKnotMap[knot];
	} else {
		knot_interval = mKnotMap[knot + 1] - mKnotMap[knot];
	}
	t = (t - mKnotMap[knot]) / knot_interval;
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
// void
// MultilevelSpline::
// ConvertMotionToSpline(std::vector<Motion*> motions, int numLevels) {
// }
// std::vector<Motion*> 
// MultilevelSpline::
// ConvertSplineToMotion() {
	
// }
}
