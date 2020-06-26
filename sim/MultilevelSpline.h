#ifndef __DEEP_PHYSICS_MULTILEVEL_SPLINE_H__
#define __DEEP_PHYSICS_MULTILEVEL_SPLINE_H__

#include <vector>
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <iostream>

namespace DPhy
{
class Spline
{
public:
	Spline(std::vector<double> knots, double end);
	Spline(double knot_interval, double end);
	void Approximate(std::vector<std::pair<Eigen::VectorXd,double>> motion);
	Eigen::VectorXd GetPosition(double t);
	std::vector<Eigen::VectorXd> GetControlPoints() { return mControlPoints; };
	void Save(std::string path);
protected:
	double B(int idx, double t);

	double mEnd;
	std::vector<double> mKnots;
	std::vector<double> mKnotMap;
	std::vector<Eigen::VectorXd> mControlPoints;
};
class MultilevelSpline
{
public:
	MultilevelSpline() {};
//	void ConvertMotionToSpline(std::vector<*> motions, int numLevels);
//	std::vector<Motion*> ConvertSplineToMotion();
protected:
//	std::vector<Motion*> mBaseMotions;
	std::vector<Spline*> mSplines;
	int mNumLevels;
};
}

#endif