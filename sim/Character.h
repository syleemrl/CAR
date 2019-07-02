#ifndef __DEEP_PHYSICS_CHARACTER_H__
#define __DEEP_PHYSICS_CHARACTER_H__
#include "dart/dart.hpp"
#include "BVH.h"
namespace DPhy
{
/**
*
* @brief Character Class
* @details Character
* 
*/
class Character
{
public:
	Character(){}
	Character(const std::string& path);
	Character(const dart::dynamics::SkeletonPtr& skeleton);

	const dart::dynamics::SkeletonPtr& GetSkeleton();

	void SetPDParameters(double kp, double kv);
	void SetPDParameters(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv);
	void SetPDParameters(const Eigen::VectorXd& k);
	void ApplyForces(const Eigen::VectorXd& forces);
	Eigen::VectorXd GetPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired);
	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired);

	void LoadBVHMap(const std::string& path);

	void InitializeBVH(BVH* bvh);
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetTargetPositionsAndVelocitiesFromBVH(BVH* bvh,double t);
	Eigen::VectorXd GetTargetPositions(BVH* bvh,double t);
protected:
	dart::dynamics::SkeletonPtr mSkeleton;

	std::map<std::string,std::string> mBVHMap; //body_node name and bvh_node name
	Eigen::VectorXd mKp, mKv;
	Eigen::VectorXd mKp_default, mKv_default;

};
};



#endif