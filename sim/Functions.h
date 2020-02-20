#ifndef __DEEP_PHYSICS_FUNCTIONS_H__
#define __DEEP_PHYSICS_FUNCTIONS_H__
#include "dart/dart.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;

namespace DPhy
{

//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<float>& val);
//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<double>& val);
//always return 1-dim array
np::ndarray toNumPyArray(const std::vector<bool>& val);
//always return 1-dim array
np::ndarray toNumPyArray(const Eigen::VectorXd& vec);
//always return 2-dim array
np::ndarray toNumPyArray(const Eigen::MatrixXd& matrix);
//always return 2-dim array
np::ndarray toNumPyArray(const std::vector<Eigen::VectorXd>& matrix);
//always return 2-dim array
np::ndarray toNumPyArray(const std::vector<std::vector<double>>& matrix);
Eigen::VectorXd toEigenVector(const np::ndarray& array,int n);
Eigen::VectorXd toEigenVector(const p::object& array,int n);
Eigen::MatrixXd toEigenMatrix(const np::ndarray& array,int n,int m);
// Utilities
std::vector<double> split_to_double(const std::string& input, int num);
    std::vector<double> split_to_double(const std::string& input);
Eigen::Vector3d string_to_vector3d(const std::string& input);
Eigen::VectorXd string_to_vectorXd(const std::string& input, int n);
    Eigen::VectorXd string_to_vectorXd(const std::string& input);
Eigen::Matrix3d string_to_matrix3d(const std::string& input);

double exp_of_squared(const Eigen::VectorXd& vec,double sigma = 1.0);
double exp_of_squared(const Eigen::Vector3d& vec,double sigma = 1.0);
double exp_of_squared(const Eigen::MatrixXd& mat,double sigma = 1.0);
std::pair<int, double> maxCoeff(const Eigen::VectorXd& in);

double RadianClamp(double input);
std::vector<dart::dynamics::BodyNode*> GetChildren(const dart::dynamics::SkeletonPtr& skel, const dart::dynamics::BodyNode* parent);

Eigen::Quaterniond DARTPositionToQuaternion(Eigen::Vector3d in);
Eigen::Vector3d QuaternionToDARTPosition(const Eigen::Quaterniond& in);
void QuaternionNormalize(Eigen::Quaterniond& in);
Eigen::VectorXd BlendPosition(Eigen::VectorXd v_target, Eigen::VectorXd v_source, double weight, bool blend_rootpos=true);
Eigen::Vector3d NearestOnGeodesicCurve3d(Eigen::Vector3d targetAxis, Eigen::Vector3d targetPosition, Eigen::Vector3d position);
Eigen::VectorXd NearestOnGeodesicCurve(Eigen::VectorXd targetAxis, Eigen::VectorXd targetPosition, Eigen::VectorXd position);
Eigen::VectorXd RotatePosition(Eigen::VectorXd pos, Eigen::VectorXd rot);
Eigen::Vector3d JointPositionDifferences(Eigen::Vector3d q2, Eigen::Vector3d q1);

void SetBodyNodeColors(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& color);
void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector3d& color);
void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector4d& color);

void EditBVH(std::string& path);
Eigen::Quaterniond GetYRotation(Eigen::Quaterniond q);

Eigen::Vector3d changeToRNNPos(Eigen::Vector3d pos);
Eigen::Isometry3d getJointTransform(dart::dynamics::SkeletonPtr skel, std::string bodyname);
Eigen::Vector4d rootDecomposition(dart::dynamics::SkeletonPtr skel, Eigen::VectorXd positions);
Eigen::VectorXd solveIK(dart::dynamics::SkeletonPtr skel, const std::string& bodyname, const Eigen::Vector3d& delta,  const Eigen::Vector3d& offset);
Eigen::VectorXd solveMCIK(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints);
Eigen::Matrix3d projectToXZ(Eigen::Matrix3d m);
Eigen::Vector3d projectToXZ(Eigen::Vector3d v);

}

#endif