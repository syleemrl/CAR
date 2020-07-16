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
	void SetKnots(std::vector<double> knots);
	void SetKnots(double knot_interval);
	std::vector<double> GetKnots() { return mKnots; };
	void Approximate(std::vector<std::pair<Eigen::VectorXd,double>> motion);
	Eigen::VectorXd GetPosition(double t);
	std::vector<Eigen::VectorXd> GetControlPoints() { return mControlPoints; };
	void SetControlPoints(std::vector<Eigen::VectorXd> cps) { mControlPoints = cps; }; 
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
	MultilevelSpline(int level, double end);
	~MultilevelSpline();
	void SetKnots(int i, std::vector<double> knots);
	void SetKnots(int i, double knot_interval);
	std::vector<double> GetKnots(int i) { return mSplines[i]->GetKnots(); };
	void ConvertMotionToSpline(std::vector<std::pair<Eigen::VectorXd,double>> motion);
	void SetControlPoints(int i, std::vector<Eigen::VectorXd> cps);
	std::vector<Eigen::VectorXd> GetControlPoints(int i);
	std::vector<Eigen::VectorXd> ConvertSplineToMotion();
protected:
	double mEnd;
	std::vector<Spline*> mSplines;
	int mNumLevels;
};
}

#endif