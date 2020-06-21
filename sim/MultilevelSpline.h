#ifndef __DEEP_PHYSICS_MULTILEVEL_SPLINE_H__
#define __DEEP_PHYSICS_MULTILEVEL_SPLINE_H__

#include "Motion.h"

namespace DPhy
{
class Spline
{
public:
	Spline(std::vector<double> knots);
	void Approximate(std::vector<Eigen::VectorXd> pos);
	Eigen::VectorXd GetPosition(double t);
protected:
	double B(int idx, int t);

	std::vector<double> mKnots;
	std::vector<Eigen::VectorXd> mControlPoints;
	std::vector<Eigen::VectorXd> mPositions;
};
class MultilevelSpline
{
public:
	MultilevelSpline() {};
	void ConvertMotionToSpline(std::vector<Motion*> motions, int numLevels);
	std::vector<Motion*> ConvertSplineToMotion();
protected:
	std::vector<Motion*> mBaseMotions;
	std::vector<Spline*> mSplines;
	int mNumLevels;
};
}

#endif