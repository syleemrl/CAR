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
class Frame
{
public:
	Frame(Eigen::VectorXd pos, Eigen::VectorXd vel) {
		position = pos;
		velocity = vel;
	}
	Frame(Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd con) {
		position = pos;
		velocity = vel;
		contact = con;
	}
	void SetPosition(Eigen::VectorXd pos) { position = pos; }
	void SetVelocity(Eigen::VectorXd vel) { velocity = vel; }
	void SetContact(Eigen::VectorXd con) { contact = con; }
	
	Eigen::VectorXd position;
	Eigen::VectorXd velocity;
	Eigen::VectorXd contact;
};
class Character
{
public:
	Character(){}
	Character(const std::string& path);
	Character(const dart::dynamics::SkeletonPtr& skeleton);

	const dart::dynamics::SkeletonPtr& GetSkeleton();
	void SetSkeleton(dart::dynamics::SkeletonPtr skel);
	void SetPDParameters(double kp, double kv);
	void SetPDParameters(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv);
	void SetPDParameters(const Eigen::VectorXd& k);
	void ApplyForces(const Eigen::VectorXd& forces);
	Eigen::VectorXd GetPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired);
	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired, const Eigen::VectorXd& v_desired);

	void LoadBVHMap(const std::string& path);

	void ReadFramesFromBVH(BVH* bvh);
	Frame* GetTargetPositionsAndVelocitiesFromBVH(BVH* bvh, double t);
	void RescaleOriginalBVH(double w);
	std::string GetContactNodeName(int i) { return mContactList[i]; };
protected:
	dart::dynamics::SkeletonPtr mSkeleton;

	std::map<std::string,std::string> mBVHMap; //body_node name and bvh_node name
	Eigen::VectorXd mKp, mKv;
	Eigen::VectorXd mKp_default, mKv_default;
	std::vector<Frame*> mBVHFrames;
	std::vector<std::string> mContactList;
};
};



#endif