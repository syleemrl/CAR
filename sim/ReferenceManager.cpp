#include <tuple>

#include "ReferenceManager.h"
#include "CharacterConfigurations.h"

namespace DPhy
{
ReferenceManager::
ReferenceManager(Character* character, double ctrl_hz)
: mCharacter(character), mControlHz(ctrl_hz), mMotionHz(30), mNumTotalFrame(0)
{
	if(this->mCharacter == nullptr){
		std::cout << "Character is null" << std::endl;
	}
	this->clear();
}

void 
ReferenceManager::
clear(){
	this->mNumTotalFrame = 0;
	this->mReferenceTrajectory.clear();
	this->mReferenceTrajectory.resize(128);
	this->mFootContactTrajectory.clear();
	this->mFootContactTrajectory.resize(128);

	int dof = this->mCharacter->GetSkeleton()->getNumDofs();
	this->mCurrentReferencePosition = Eigen::VectorXd::Zero(dof);
	this->mNextReferencePosition = Eigen::VectorXd::Zero(dof);
	this->mCurrentReferenceVelocity = Eigen::VectorXd::Zero(dof);
}

void 
ReferenceManager::
addPosition(const Eigen::VectorXd& ref_pos){
	double ik_height = 0.04;
	if( this->mNumTotalFrame >= this->mReferenceTrajectory.size() )
	{
		this->mReferenceTrajectory.resize(2*this->mReferenceTrajectory.size());
		this->mFootContactTrajectory.resize(2*this->mFootContactTrajectory.size());
	}
	int i = this->mNumTotalFrame;
	this->mNumTotalFrame += 1;

	this->mReferenceTrajectory[i] = this->convertRNNMotion(ref_pos);
	this->mFootContactTrajectory[i] = ref_pos.segment<2>(FOOT_CONTACT_OFFSET);

	// solve ik for foot contact
	
	auto skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();
	int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);

	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);

	// // if left foot contacting
	// // pull left foot to ground
	// if(this->mFootContactTrajectory[i][0] > FOOT_CONTACT_THERESHOLD){
	// 	skel->setPositions(this->mReferenceTrajectory[i]);
	// 	skel->computeForwardKinematics(true, false, false);
	// 	auto ret = this->getFootIKConstraints(0, 1);
	// 	if(ret.size()){
	// 		this->mReferenceTrajectory[i][tibia_l_idx] = 0.1*M_PI;
	// 		skel->setPositions(this->mReferenceTrajectory[i]);
	// 		skel->computeForwardKinematics(true, false, false);
	// 		Eigen::VectorXd newPose = solveMCIK(skel, ret);
	// 		this->mReferenceTrajectory[i] = newPose;
	// 	}
	// }

	// // if right foot contacting
	// // pull right foot to ground
	// if(this->mFootContactTrajectory[i][1] > FOOT_CONTACT_THERESHOLD){
	// 	skel->setPositions(this->mReferenceTrajectory[i]);
	// 	skel->computeForwardKinematics(true, false, false);
	// 	auto ret = this->getFootIKConstraints(1, 1);
	// 	if(ret.size()){
	// 		this->mReferenceTrajectory[i][tibia_r_idx] = 0.1*M_PI;
	// 		skel->setPositions(this->mReferenceTrajectory[i]);
	// 		skel->computeForwardKinematics(true, false, false);
	// 		Eigen::VectorXd newPose = solveMCIK(skel, ret);
	// 		this->mReferenceTrajectory[i] = newPose;
	// 	}
	// }

	// preventing foot penetration
	// skel->setPositions(this->mReferenceTrajectory[i]);
	// skel->computeForwardKinematics(true, false, false);
	// auto ret = this->getFootIKConstraints(0, 0);
	// if(ret.size()){
	// 	this->mReferenceTrajectory[i][tibia_l_idx] = 0.1*M_PI;
	// 	skel->setPositions(this->mReferenceTrajectory[i]);
	// 	skel->computeForwardKinematics(true, false, false);
	// 	Eigen::VectorXd newPose = solveMCIK(skel, ret);
	// 	this->mReferenceTrajectory[i] = newPose;
	// }

	// skel->setPositions(this->mReferenceTrajectory[i]);
	// skel->computeForwardKinematics(true, false, false);
	// ret = this->getFootIKConstraints(1, 0);
	// if(ret.size()){
	// 	this->mReferenceTrajectory[i][tibia_r_idx] = 0.1*M_PI;
	// 	skel->setPositions(this->mReferenceTrajectory[i]);
	// 	skel->computeForwardKinematics(true, false, false);
	// 	Eigen::VectorXd newPose = solveMCIK(skel, ret);
	// 	this->mReferenceTrajectory[i] = newPose;
	// }
	// skel->setPositions(p_save);
	// skel->computeForwardKinematics(true, false, false);
}

Eigen::VectorXd
ReferenceManager::
getPositions(double time){
	if(this->mNumTotalFrame == 0){
		return this->mCurrentReferencePosition;
		// std::cout << "Reference trajectory is not set" << std::endl;
		// exit(0);
	}
	int k = (int)std::floor(time*this->mMotionHz)%this->mNumTotalFrame;
	int k1 = std::min(k+1,this->mNumTotalFrame-1);
	// double t = std::fmod(time, (1.0/this->mMotionHz))*this->mMotionHz;
	double t = (time*this->mMotionHz-k);
	if( t < 0 )
		std::cout << time << " : " << k << ", " << k1 << ", " << t << std::endl;

	Eigen::VectorXd motion_k = this->mReferenceTrajectory[k];
	Eigen::VectorXd motion_k1 = this->mReferenceTrajectory[k1];

	Eigen::VectorXd motion_t = Eigen::VectorXd::Zero(motion_k.rows());

	auto& skel = this->mCharacter->GetSkeleton();
	for(int i = 0; i < skel->getNumJoints(); i++){
		dart::dynamics::Joint* jn = skel->getJoint(i);
		if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr){
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			Eigen::Quaterniond pos_k = DARTPositionToQuaternion(motion_k.segment<3>(index));
			Eigen::Quaterniond pos_k1 = DARTPositionToQuaternion(motion_k1.segment<3>(index));

			motion_t.segment<3>(index) = QuaternionToDARTPosition(pos_k.slerp(t, pos_k1));
		}
		else if(dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			Eigen::Quaterniond pos_k = DARTPositionToQuaternion(motion_k.segment<3>(index));
			Eigen::Quaterniond pos_k1 = DARTPositionToQuaternion(motion_k1.segment<3>(index));

			motion_t.segment<3>(index) = QuaternionToDARTPosition(pos_k.slerp(t, pos_k1));

			motion_t.segment<3>(index+3) = motion_k.segment<3>(index+3)*(1-t) + motion_k1.segment<3>(index+3)*t;
		}
		else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			double delta = RadianClamp(motion_k1[index]-motion_k[index]);
			motion_t[index] = motion_k[index] + delta*t;
		}
	}

	return motion_t;
}

Eigen::VectorXd
ReferenceManager::
getPositions(int count){
	if(this->mNumTotalFrame == 0){
		return this->mCurrentReferencePosition;
		// std::cout << "Reference trajectory is not set" << std::endl;
		// exit(0);
	}
	if(count >= this->mNumTotalFrame){
		std::cout << "ReferenceManager.cpp : count exceeds frame limit" << std::endl;
		std::cout << "count : " << count << ", " << "tf : " << mNumTotalFrame << std::endl;
		count = this->mNumTotalFrame-1;
	}
	return this->mReferenceTrajectory[count];
}

Eigen::VectorXd
ReferenceManager::
getPositionsAndVelocities(double time){
    double next_time = time + 1.0/this->mControlHz;
    Eigen::VectorXd cur_pos = this->getPositions(time);
    Eigen::VectorXd next_pos = this->getPositions(next_time);
    Eigen::VectorXd cur_vel = this->mCharacter->GetSkeleton()->getPositionDifferences(next_pos, cur_pos) * this->mControlHz;

	Eigen::VectorXd motion_t = Eigen::VectorXd::Zero(cur_pos.rows() * 2);
	motion_t << cur_pos, cur_vel;
	return motion_t;
}

Eigen::VectorXd
ReferenceManager::
getPositionsAndVelocities(int count){
    int next_count = count + 1;
    Eigen::VectorXd cur_pos = this->getPositions(count);
    Eigen::VectorXd next_pos = this->getPositions(next_count);
    Eigen::VectorXd cur_vel = this->mCharacter->GetSkeleton()->getPositionDifferences(next_pos, cur_pos) * this->mControlHz;

	Eigen::VectorXd motion_t = Eigen::VectorXd::Zero(cur_pos.rows() * 2);
	motion_t << cur_pos, cur_vel;
	return motion_t;
}

Eigen::VectorXd
ReferenceManager::
getOriginPositions(double time){
	if(this->mNumTotalFrame == 0){
		return this->mCurrentReferencePosition;
		// std::cout << "Reference trajectory is not set" << std::endl;
		// exit(0);
	}
	int k = (int)std::floor(time*this->mMotionHz)%this->mNumTotalFrame;
	int k1 = std::min(k+1,this->mNumTotalFrame-1);
	double t = (time*this->mMotionHz-k);

	Eigen::VectorXd motion_k = this->mReferenceTrajectoryOrigin[k];
	Eigen::VectorXd motion_k1 = this->mReferenceTrajectoryOrigin[k1];

	Eigen::VectorXd motion_t = Eigen::VectorXd::Zero(motion_k.rows());

	auto& skel = this->mCharacter->GetSkeleton();
	for(int i = 0; i < skel->getNumJoints(); i++){
		dart::dynamics::Joint* jn = skel->getJoint(i);	
		if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr){	
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			Eigen::Quaterniond pos_k = DARTPositionToQuaternion(motion_k.segment<3>(index));
			Eigen::Quaterniond pos_k1 = DARTPositionToQuaternion(motion_k1.segment<3>(index));

			motion_t.segment<3>(index) = QuaternionToDARTPosition(pos_k.slerp(t, pos_k1));
		}
		else if(dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){	
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			Eigen::Quaterniond pos_k = DARTPositionToQuaternion(motion_k.segment<3>(index));
			Eigen::Quaterniond pos_k1 = DARTPositionToQuaternion(motion_k1.segment<3>(index));

			motion_t.segment<3>(index) = QuaternionToDARTPosition(pos_k.slerp(t, pos_k1));

			motion_t.segment<3>(index+3) = motion_k.segment<3>(index+3)*(1-t) + motion_k1.segment<3>(index+3)*t;
		}
		else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){	
			int index = jn->getIndexInSkeleton(0);
			int dof = jn->getNumDofs();
			double delta = RadianClamp(motion_k1[index]-motion_k[index]);
			motion_t[index] = motion_k[index] + delta*t;
		}
	}

	return motion_t;
}

Eigen::VectorXd
ReferenceManager::
getOriginPositions(int count){
	if(this->mNumTotalFrame == 0){
		return this->mCurrentReferencePosition;
		// std::cout << "Reference trajectory is not set" << std::endl;
		// exit(0);
	}
	if(count >= this->mNumTotalFrame){
		std::cout << "ReferenceManager.cpp : count exceeds frame limit" << std::endl;
		count = this->mNumTotalFrame-1;
	}
	return this->mReferenceTrajectoryOrigin[count];
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
ReferenceManager::
getMotion(double time)
{
	if(this->mNumTotalFrame == 0){
		return this->getMotion();
	}
	if( time > this->getMaxTime()){
		std::cout << "Exceed reference max time" << std::endl;
		Eigen::VectorXd cur_pos = this->getPositions(time);
		Eigen::VectorXd next_pos = this->getPositions(time);
		Eigen::VectorXd cur_vel = Eigen::VectorXd::Zero(cur_pos.rows());
		return std::make_tuple(cur_pos, cur_vel, next_pos);
	}

	double next_time = time + 1.0/this->mControlHz;
	Eigen::VectorXd cur_pos = this->getPositions(time);
	Eigen::VectorXd next_pos = this->getPositions(next_time);
	Eigen::VectorXd cur_vel = this->mCharacter->GetSkeleton()->getPositionDifferences(next_pos, cur_pos) * this->mControlHz;

	return std::make_tuple(cur_pos, cur_vel, next_pos);
}


std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
ReferenceManager::
getMotion(int count)
{
	if(this->mNumTotalFrame == 0){
		return this->getMotion();
	}
	if( count > this->getMaxCount()){
		std::cout << "Exceed reference max time" << std::endl;
		Eigen::VectorXd cur_pos = this->getPositions(count);
		Eigen::VectorXd next_pos = this->getPositions(count);
		Eigen::VectorXd cur_vel = Eigen::VectorXd::Zero(cur_pos.rows());
		return std::make_tuple(cur_pos, cur_vel, next_pos);
	}

	int next_count = count + 1;
	Eigen::VectorXd cur_pos = this->getPositions(count);
	Eigen::VectorXd next_pos = this->getPositions(next_count);
	Eigen::VectorXd cur_vel = this->mCharacter->GetSkeleton()->getPositionDifferences(next_pos, cur_pos) * this->mControlHz;

	return std::make_tuple(cur_pos, cur_vel, next_pos);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
ReferenceManager::
getMotion()
{
	return std::make_tuple(this->mCurrentReferencePosition, 
							this->mCurrentReferenceVelocity, 
							this->mNextReferencePosition);
}

Eigen::Vector2d
ReferenceManager::
getFootContacts(double time)
{
	if(this->mNumTotalFrame == 0){
		return Eigen::Vector2d::Constant(-1);
	}
	int k = (int)std::floor(time*this->mMotionHz)%this->mNumTotalFrame;
	int k1 = std::min(k+1,this->mNumTotalFrame-1);
	double t = std::fmod(time, (1.0/this->mMotionHz))*this->mMotionHz;

	Eigen::Vector2d motion_k = this->mFootContactTrajectory[k];
	Eigen::Vector2d motion_k1 = this->mFootContactTrajectory[k1];

	Eigen::Vector2d motion_t = (1-t)*motion_k + t*motion_k1;

	Eigen::Vector2d ret = Eigen::Vector2d::Constant(-1);
	if(motion_t[0] > FOOT_CONTACT_THERESHOLD){
		ret[0] = 1;
	}
	if(motion_t[1] > FOOT_CONTACT_THERESHOLD){
		ret[1] = 1;
	}

	return ret;
}

Eigen::Vector2d
ReferenceManager::
getFootContacts(int count)
{
	if(this->mNumTotalFrame == 0){
		return Eigen::Vector2d::Constant(-1);
	}
	Eigen::Vector2d ret = Eigen::Vector2d::Constant(-1);
	if(this->mFootContactTrajectory[count][0] > FOOT_CONTACT_THERESHOLD){
		ret[0] = 1;
	}
	if(this->mFootContactTrajectory[count][1] > FOOT_CONTACT_THERESHOLD){
		ret[1] = 1;
	}

	return ret;
}

void
ReferenceManager::
setTargetMotion(const Eigen::VectorXd& ref_cur, const Eigen::VectorXd& ref_next)
{
	this->mCurrentReferencePosition = this->convertRNNMotion(ref_cur);
	this->mNextReferencePosition = this->convertRNNMotion(ref_next);

	this->mCurrentReferenceVelocity = this->mCharacter->GetSkeleton()->getPositionDifferences(this->mNextReferencePosition, this->mCurrentReferencePosition)*this->mMotionHz;
}


std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>
ReferenceManager::
getFootIKConstraints(bool isRight, bool isPull){

	const dart::dynamics::BodyNode *bn1, *bn2;
	std::string bn1_str, bn2_str;
	if(isRight){
		bn1_str = "FootR";
		bn2_str = "FootEndR";		
	}
	else{
		bn1_str = "FootL";
		bn2_str = "FootEndL";		
	}
	bn1 = this->mCharacter->GetSkeleton()->getBodyNode(bn1_str);
	bn2 = this->mCharacter->GetSkeleton()->getBodyNode(bn2_str);

	std::vector<Eigen::Vector3d> offsetList;
	offsetList.resize(4);

#ifdef NEW_JOINTS
	offsetList[0] = Eigen::Vector3d(0.04, -0.025, -0.065);
	offsetList[1] = Eigen::Vector3d(-0.04, -0.025, -0.065);
	offsetList[2] = Eigen::Vector3d(0.04, -0.025, 0.035);
	offsetList[3] = Eigen::Vector3d(-0.04, -0.025, 0.035);
#else
	offsetList[0] = Eigen::Vector3d(0.0375, -0.065, 0.025);
	offsetList[1] = Eigen::Vector3d(-0.0375, -0.065, 0.025);
	offsetList[2] = Eigen::Vector3d(0.0375, 0.025, 0.025);
	offsetList[3] = Eigen::Vector3d(-0.0375, 0.025, 0.025);
#endif
	std::vector<Eigen::Vector3d> tpList;
	tpList.resize(4);
	for(int i = 0 ; i < 2; i++){
		tpList[i] = bn1->getWorldTransform()*offsetList[i];
	}
	for(int i = 2 ; i < 4; i++){
		tpList[i] = bn2->getWorldTransform()*offsetList[i];
	}

	double min = tpList[0][1];
	int min_idx = 0;
	for(int i = 1; i < 4; i++){
		if(tpList[i][1] < min){
			min = tpList[i][1];
			min_idx = i;
		}
	}

	std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> ret;
	double height_threshold = 0.0;
	if(isPull){
		for(int i = 0; i < 4; i++){
			tpList[i][1] = 0;
		}
		ret.push_back(std::make_tuple(bn1_str, tpList[0], offsetList[0]));
		ret.push_back(std::make_tuple(bn1_str, tpList[1], offsetList[1]));
		ret.push_back(std::make_tuple(bn2_str, tpList[2], offsetList[2]));
		ret.push_back(std::make_tuple(bn2_str, tpList[3], offsetList[3]));
	}
	else{
		if(min < -height_threshold){
			double delta = height_threshold + min;
			for(int i = 0; i < 4;  i++){
				tpList[i][1] = std::max(tpList[i][1], 0.0);
			}
			ret.push_back(std::make_tuple(bn1_str, tpList[0], offsetList[0]));
			ret.push_back(std::make_tuple(bn1_str, tpList[1], offsetList[1]));
			ret.push_back(std::make_tuple(bn2_str, tpList[2], offsetList[2]));
			ret.push_back(std::make_tuple(bn2_str, tpList[3], offsetList[3]));
		}

	}
	return ret;
}

// HS) given trajectory is come from X,Y.dat.
//     those are not proper to physics reference.
//     so, this function solving IK for total trajectory.
void 
ReferenceManager::
setTrajectory(const Eigen::MatrixXd& trajectory){
	double ik_height = 0.04;
	this->mNumTotalFrame = trajectory.rows();
	this->mReferenceTrajectory.clear();
	this->mReferenceTrajectory.resize(this->mNumTotalFrame);
	this->mReferenceTrajectoryOrigin.clear();
	this->mReferenceTrajectoryOrigin.resize(this->mNumTotalFrame);
	this->mFootContactTrajectory.clear();
	this->mFootContactTrajectory.resize(this->mNumTotalFrame);
	for(int i = 0; i < this->mNumTotalFrame; i++){
		this->mReferenceTrajectory[i] = this->convertRNNMotion(trajectory.row(i));
		this->mReferenceTrajectoryOrigin[i] = this->mReferenceTrajectory[i];
		this->mFootContactTrajectory[i] = trajectory.block<1,2>(i,FOOT_CONTACT_OFFSET);


		auto skel = this->mCharacter->GetSkeleton();
		int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
		int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);

		int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
		int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);


		// if left foot contacting
		// pull left foot to ground
		// if(this->mFootContactTrajectory[i][0] > FOOT_CONTACT_THERESHOLD){

		// 	skel->setPositions(this->mReferenceTrajectory[i]);
		// 	skel->computeForwardKinematics(true, false, false);
		// 	auto ret = this->getFootIKConstraints(0, 1);
		// 	if(ret.size()){
		// 		this->mReferenceTrajectory[i][tibia_l_idx] = 0.1*M_PI;
		// 		skel->setPositions(this->mReferenceTrajectory[i]);
		// 		skel->computeForwardKinematics(true, false, false);
		// 		Eigen::VectorXd newPose = solveMCIK(skel, ret);
		// 		this->mReferenceTrajectory[i] = newPose;
		// 	}
		// }

		// // if right foot contacting
		// // pull right foot to ground
		// if(this->mFootContactTrajectory[i][1] > FOOT_CONTACT_THERESHOLD){
		// 	skel->setPositions(this->mReferenceTrajectory[i]);
		// 	skel->computeForwardKinematics(true, false, false);
		// 	auto ret = this->getFootIKConstraints(1, 1);
		// 	if(ret.size()){
		// 		this->mReferenceTrajectory[i][tibia_r_idx] = 0.1*M_PI;
		// 		skel->setPositions(this->mReferenceTrajectory[i]);
		// 		skel->computeForwardKinematics(true, false, false);
		// 		Eigen::VectorXd newPose = solveMCIK(skel, ret);
		// 		this->mReferenceTrajectory[i] = newPose;
		// 	}
		// }

		// preventing foot penetration
		// skel->setPositions(this->mReferenceTrajectory[i]);
		// skel->computeForwardKinematics(true, false, false);
		// auto ret = this->getFootIKConstraints(0, 0);
		// if(ret.size()){
		// 	this->mReferenceTrajectory[i][tibia_l_idx] = 0.1*M_PI;
		// 	skel->setPositions(this->mReferenceTrajectory[i]);
		// 	skel->computeForwardKinematics(true, false, false);
		// 	Eigen::VectorXd newPose = solveMCIK(skel, ret);
		// 	this->mReferenceTrajectory[i] = newPose;
		// }

		// skel->setPositions(this->mReferenceTrajectory[i]);
		// skel->computeForwardKinematics(true, false, false);
		// ret = this->getFootIKConstraints(1, 0);
		// if(ret.size()){
		// 	this->mReferenceTrajectory[i][tibia_r_idx] = 0.1*M_PI;
		// 	skel->setPositions(this->mReferenceTrajectory[i]);
		// 	skel->computeForwardKinematics(true, false, false);
		// 	Eigen::VectorXd newPose = solveMCIK(skel, ret);
		// 	this->mReferenceTrajectory[i] = newPose;
		// }
	}
}

void ReferenceManager::saveReferenceTrajectory(std::string filename) {
    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@saveReferenceTrajectory!" << std::endl;
    FILE *out = std::fopen((std::string(DPHY_DIR) + "/learned_trajectory/" + filename).c_str(),"w");
    for(int i = 0; i < this->mNumTotalFrame; i++){
        for (int j = 0; j < this->mReferenceTrajectory[i].size(); j++){
            fprintf(out,"%lf ", this->mReferenceTrajectory[i][j]);
        }
        fprintf(out,"\n");
    }
    fclose(out);
}

void 
ReferenceManager::
setGoalTrajectory(const std::vector<Eigen::Vector3d>& goal_trajectory)
{
	if(this->mNumTotalFrame != goal_trajectory.size()){
		std::cout << "mismatch between the number of motion and goal frames" << std::endl;
		exit(0);
	}
	this->mGoalTrajectory.clear();
	this->mGoalTrajectory.resize(this->mNumTotalFrame);
	for(int i = 0; i < this->mNumTotalFrame; i++){
		this->mGoalTrajectory[i] = goal_trajectory[i];
	}
}
void ReferenceManager::saveGoalTrajectory(std::string filename) {
	std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@saveGoalTrajectory!" << std::endl;
	FILE *out = std::fopen((std::string(DPHY_DIR) + "/learned_trajectory/" + filename).c_str(),"w");
	for(int i = 0; i < this->mNumTotalFrame; i++){
		for (int j = 0; j < 3; j++){
			fprintf(out,"%lf ", this->mGoalTrajectory[i][j]);
		}
		fprintf(out,"\n");
	}
	fclose(out);
}

void 
ReferenceManager::
setGoal(const Eigen::Vector3d& goal)
{
	this->mGoal = goal;
}

Eigen::Vector3d 
ReferenceManager::
getGoal()
{
	return this->mGoal;
}

Eigen::Vector3d 
ReferenceManager::
getGoal(double time)
{
	if(this->mNumTotalFrame == 0){
		return this->getGoal();
	}
	int k = (int)std::floor(time*this->mMotionHz)%this->mNumTotalFrame;
	return this->mGoalTrajectory[k];
}

Eigen::Vector3d 
ReferenceManager::
getGoal(int count)
{
	if(this->mNumTotalFrame == 0){
		return this->getGoal();
	}
	return this->mGoalTrajectory[count];
}

Eigen::Quaterniond
ReferenceManager::
convertRNNMotion(const Eigen::VectorXd& ref, std::string bodyName, int index_in_input, const Eigen::Quaterniond& offset){
	Eigen::Quaterniond q = DARTPositionToQuaternion(offset._transformVector(ref.segment<3>(index_in_input)));
	if(bodyName == "TalusL" || bodyName == "TalusR" ){
		q = Eigen::AngleAxisd(FOOT_OFFSET*M_PI, Eigen::Vector3d::UnitX())*q;
	}
	return q;
}

void 
ReferenceManager::
setRNNMotion(Eigen::VectorXd& converted_motion, const Eigen::Quaterniond& rnn_motion, std::string bodyName){
	int idx = this->mCharacter->GetSkeleton()->getBodyNode(bodyName)->getParentJoint()->getIndexInSkeleton(0);
	converted_motion.segment<3>(idx) = QuaternionToDARTPosition(rnn_motion);
}

void 
ReferenceManager::
setRNNMotion(Eigen::VectorXd& converted_motion, const Eigen::Vector3d& rnn_motion, std::string bodyName){
	int idx = this->mCharacter->GetSkeleton()->getBodyNode(bodyName)->getParentJoint()->getIndexInSkeleton(0);
	converted_motion.segment<3>(idx) = rnn_motion;
}

void
ReferenceManager::
convertAndSetRNNMotion(Eigen::VectorXd& converted_motion, const Eigen::VectorXd& ref, std::string bodyName, int index_in_input, const Eigen::Quaterniond& offset){
	Eigen::Quaterniond q = convertRNNMotion(ref, bodyName, index_in_input, offset);

	int idx = this->mCharacter->GetSkeleton()->getBodyNode(bodyName)->getParentJoint()->getIndexInSkeleton(0);
	setRNNMotion(converted_motion, q, bodyName);
}


// HS) "ref" is output of RNN (more precisely, come from rnn_manager.getReferences(target))
//	   It has to be converted into friendly hierarchy.
Eigen::VectorXd
ReferenceManager::
convertRNNMotion(const Eigen::VectorXd& ref)
{
	Eigen::VectorXd converted_motion = this->mCharacter->GetSkeleton()->getPositions();
	converted_motion.setZero();

	// root position
	converted_motion[3] = -ref[2]*0.01;
	converted_motion[4] = ref[1]*0.01 + ROOT_HEIGHT_OFFSET;
	converted_motion[5] = ref[0]*0.01;

#ifdef ALL_JOINTS
	// ori offset
	Eigen::Matrix3d mat_offset_hip;
	mat_offset_hip << 0, 0, 1,
					  1, 0, 0,
					  0, 1, 0;
	Eigen::Quaterniond q_offset_hip(mat_offset_hip);

	Eigen::Matrix3d mat_offset_arm_l;
	mat_offset_arm_l << 1, 0, 0,
						0, 0, 1,
						0, -1, 0;
	Eigen::Quaterniond q_offset_arm_l(mat_offset_arm_l);

	Eigen::Matrix3d mat_offset_hand_l;
	mat_offset_hand_l << 1, 0, 0,
						0, -1, 0,
						0, 0, -1;
	Eigen::Quaterniond q_offset_hand_l(mat_offset_hand_l);

	Eigen::Matrix3d mat_offset_arm_r;
	mat_offset_arm_r << -1, 0, 0,
						0, 0, -1,
						0, -1, 0;
	Eigen::Quaterniond q_offset_arm_r(mat_offset_arm_r);

	Eigen::Matrix3d mat_offset_hand_r;
	mat_offset_hand_r << -1, 0, 0,
						0, -1, 0,
						0, 0, 1;
	Eigen::Quaterniond q_offset_hand_r(mat_offset_hand_r);

	Eigen::Matrix3d mat_offset_leg;
	mat_offset_leg << 0, 0, -1,
					  -1, 0, 0,
					  0, 1, 0;
	Eigen::Quaterniond q_offset_leg(mat_offset_leg);

	// root orientation
	// std::cout << ref[3] << std::endl;
	Eigen::Quaterniond root_y_ori(Eigen::AngleAxisd(ref[3], Eigen::Vector3d::UnitY()));
	Eigen::Quaterniond hip_ori = DARTPositionToQuaternion(q_offset_hip._transformVector(ref.segment<3>(4)));
	root_y_ori = root_y_ori * hip_ori;
	converted_motion.segment<3>(0) = QuaternionToDARTPosition(root_y_ori);

	int idx;
	convertAndSetRNNMotion(converted_motion, ref, "FemurL", 7, q_offset_leg);
	convertAndSetRNNMotion(converted_motion, ref, "TibiaL", 10, q_offset_leg);
	
	Eigen::Quaterniond qTalusL = convertRNNMotion(ref, "TalusL", 13, q_offset_leg);
	Eigen::Quaterniond qFootL = convertRNNMotion(ref, "FootL", 16, q_offset_leg);
	setRNNMotion(converted_motion, qTalusL*qFootL, "FootL");

	convertAndSetRNNMotion(converted_motion, ref, "FemurR", 19, q_offset_leg);
	convertAndSetRNNMotion(converted_motion, ref, "TibiaR", 22, q_offset_leg);
	
	Eigen::Quaterniond qTalusR = convertRNNMotion(ref, "TalusR", 25, q_offset_leg);
	Eigen::Quaterniond qFootR = convertRNNMotion(ref, "FootR", 28, q_offset_leg);
	setRNNMotion(converted_motion, qTalusR*qFootR, "FootR");

	Eigen::Quaterniond qSpine = convertRNNMotion(ref, "Spine", 31, q_offset_hip);
	Eigen::Quaterniond qSpine1 = convertRNNMotion(ref, "Spine1", 34, q_offset_hip);
	Eigen::Quaterniond qSpine2 = convertRNNMotion(ref, "Spine2", 37, q_offset_hip);
	setRNNMotion(converted_motion, qSpine*qSpine1*qSpine2, "Spine1");	

	Eigen::Quaterniond qNeck = convertRNNMotion(ref, "Neck", 40, q_offset_hip);
	Eigen::Quaterniond qHead = convertRNNMotion(ref, "Head", 43, q_offset_hip);
	setRNNMotion(converted_motion, qNeck*qHead, "Head");	

	Eigen::Quaterniond qShoulderL = convertRNNMotion(ref, "ShoulderL", 46, q_offset_arm_l);
	Eigen::Quaterniond qArmL = convertRNNMotion(ref, "ArmL", 49, q_offset_arm_l);
	setRNNMotion(converted_motion, qNeck*qShoulderL*qArmL, "ArmL");	

	convertAndSetRNNMotion(converted_motion, ref, "ForeArmL", 52, q_offset_arm_l);

	convertAndSetRNNMotion(converted_motion, ref, "HandL", 55, q_offset_hand_l);

	Eigen::Quaterniond qShoulderR = convertRNNMotion(ref, "ShoulderR", 58, q_offset_arm_r);
	Eigen::Quaterniond qArmR = convertRNNMotion(ref, "ArmR", 61, q_offset_arm_r);
	setRNNMotion(converted_motion, qNeck*qShoulderR*qArmR, "ArmR");	

	convertAndSetRNNMotion(converted_motion, ref, "ForeArmR", 64, q_offset_arm_r);

	convertAndSetRNNMotion(converted_motion, ref, "HandR", 67, q_offset_hand_r);
#endif
#ifdef CMU_JOINTS
	// ori offset
	Eigen::Matrix3d mat_offset_hip;
	mat_offset_hip << 0, 0, 1,
					  1, 0, 0,
					  0, 1, 0;
	mat_offset_hip = Eigen::Matrix3d::Identity();
	Eigen::Quaterniond q_offset_hip(mat_offset_hip);

	Eigen::Matrix3d mat_offset_arm_l;
	mat_offset_arm_l << 1, 0, 0,
						0, 0, 1,
						0, -1, 0;
    mat_offset_arm_l = Eigen::Matrix3d::Identity();
	Eigen::Quaterniond q_offset_arm_l(mat_offset_arm_l);

	Eigen::Matrix3d mat_offset_hand_l;
	mat_offset_hand_l << 1, 0, 0,
						0, -1, 0,
						0, 0, -1;
    mat_offset_hand_l = Eigen::Matrix3d::Identity();
	Eigen::Quaterniond q_offset_hand_l(mat_offset_hand_l);

	Eigen::Matrix3d mat_offset_arm_r;
	mat_offset_arm_r << -1, 0, 0,
						0, 0, -1,
						0, -1, 0;
    mat_offset_arm_r = Eigen::Matrix3d::Identity();
	Eigen::Quaterniond q_offset_arm_r(mat_offset_arm_r);

	Eigen::Matrix3d mat_offset_hand_r;
	mat_offset_hand_r << -1, 0, 0,
						0, -1, 0,
						0, 0, 1;
    mat_offset_hand_r = Eigen::Matrix3d::Identity();
	Eigen::Quaterniond q_offset_hand_r(mat_offset_hand_r);

	Eigen::Matrix3d mat_offset_leg;
	mat_offset_leg << 0, 0, -1,
					  -1, 0, 0,
					  0, 1, 0;
    mat_offset_leg = Eigen::Matrix3d::Identity();
	Eigen::Quaterniond q_offset_leg(mat_offset_leg);

	// root orientation
	// std::cout << ref[3] << std::endl;
	Eigen::Quaterniond root_y_ori(Eigen::AngleAxisd(ref[3], Eigen::Vector3d::UnitY()));
	Eigen::Quaterniond hip_ori = DARTPositionToQuaternion(q_offset_hip._transformVector(ref.segment<3>(4)));
	root_y_ori = root_y_ori * hip_ori;
	converted_motion.segment<3>(0) = QuaternionToDARTPosition(root_y_ori);

	int idx;
	Eigen::Quaterniond qLHip = convertRNNMotion(ref, "LHip", 72, q_offset_leg);
	Eigen::Quaterniond qFemurL = convertRNNMotion(ref, "FemurL", 7, q_offset_leg);
	setRNNMotion(converted_motion, qLHip*qFemurL, "FemurL");

	convertAndSetRNNMotion(converted_motion, ref, "TibiaL", 10, q_offset_leg);
	
	Eigen::Quaterniond qTalusL = convertRNNMotion(ref, "TalusL", 13, q_offset_leg);
	Eigen::Quaterniond qFootL = convertRNNMotion(ref, "FootL", 16, q_offset_leg);
	setRNNMotion(converted_motion, qTalusL, "FootL");

	Eigen::Quaterniond qRHip = convertRNNMotion(ref, "RHip", 75, q_offset_leg);
	Eigen::Quaterniond qFemurR = convertRNNMotion(ref, "FemurR", 19, q_offset_leg);
	setRNNMotion(converted_motion, qRHip*qFemurR, "FemurR");
	// convertAndSetRNNMotion(converted_motion, ref, "FemurR", 19, q_offset_leg);
	convertAndSetRNNMotion(converted_motion, ref, "TibiaR", 22, q_offset_leg);
	
	Eigen::Quaterniond qTalusR = convertRNNMotion(ref, "TalusR", 25, q_offset_leg);
	Eigen::Quaterniond qFootR = convertRNNMotion(ref, "FootR", 28, q_offset_leg);
	setRNNMotion(converted_motion, qTalusR, "FootR");

	Eigen::Quaterniond qLowerBack = convertRNNMotion(ref, "LowerBack", 31, q_offset_hip);
	Eigen::Quaterniond qSpine = convertRNNMotion(ref, "Spine", 34, q_offset_hip);
    Eigen::Quaterniond qSpine1 = convertRNNMotion(ref, "Spine1", 37, q_offset_hip);
	setRNNMotion(converted_motion, qLowerBack*qSpine*qSpine1, "Spine1");

	Eigen::Quaterniond qNeck = convertRNNMotion(ref, "Neck", 40, q_offset_hip);
	Eigen::Quaterniond qHead = convertRNNMotion(ref, "Head", 43, q_offset_hip);
	setRNNMotion(converted_motion, qNeck*qHead, "Head");	

	Eigen::Quaterniond qShoulderL = convertRNNMotion(ref, "ShoulderL", 46, q_offset_arm_l);
	Eigen::Quaterniond qArmL = convertRNNMotion(ref, "ArmL", 49, q_offset_arm_l);
	setRNNMotion(converted_motion, qShoulderL*qArmL, "ArmL");

	convertAndSetRNNMotion(converted_motion, ref, "ForeArmL", 52, q_offset_arm_l);

	convertAndSetRNNMotion(converted_motion, ref, "HandL", 55, q_offset_hand_l);

	Eigen::Quaterniond qShoulderR = convertRNNMotion(ref, "ShoulderR", 58, q_offset_arm_r);
	Eigen::Quaterniond qArmR = convertRNNMotion(ref, "ArmR", 61, q_offset_arm_r);
	setRNNMotion(converted_motion, qShoulderR*qArmR, "ArmR");

	convertAndSetRNNMotion(converted_motion, ref, "ForeArmR", 64, q_offset_arm_r);

	convertAndSetRNNMotion(converted_motion, ref, "HandR", 67, q_offset_hand_r);

#endif

#ifdef NEW_JOINTS
	// root orientation
	// std::cout << ref[3] << std::endl;
	Eigen::Quaterniond root_y_ori(Eigen::AngleAxisd(ref[3], Eigen::Vector3d::UnitY()));
	Eigen::Quaterniond hip_ori = DARTPositionToQuaternion(ref.segment<3>(4));
	root_y_ori = root_y_ori * hip_ori;
	converted_motion.segment<3>(0) = QuaternionToDARTPosition(root_y_ori);

	// converted_motion.tail(converted_motion.rows()-6) = ref.segment(7,converted_motion.rows()-6);

	setRNNMotion(converted_motion, ref.segment<3>(7), "Spine");
	setRNNMotion(converted_motion, ref.segment<3>(10), "Neck");
	setRNNMotion(converted_motion, ref.segment<3>(13), "Head");

	setRNNMotion(converted_motion, ref.segment<3>(16), "ArmL");
	setRNNMotion(converted_motion, ref.segment<3>(19), "ForeArmL");
	setRNNMotion(converted_motion, ref.segment<3>(22), "HandL");

	setRNNMotion(converted_motion, ref.segment<3>(25), "ArmR");
	setRNNMotion(converted_motion, ref.segment<3>(28), "ForeArmR");
	setRNNMotion(converted_motion, ref.segment<3>(31), "HandR");

	setRNNMotion(converted_motion, ref.segment<3>(34), "FemurL");
	setRNNMotion(converted_motion, ref.segment<3>(37), "TibiaL");
	setRNNMotion(converted_motion, ref.segment<3>(40), "FootL");

	setRNNMotion(converted_motion, ref.segment<3>(43), "FemurR");
	setRNNMotion(converted_motion, ref.segment<3>(46), "TibiaR");
	setRNNMotion(converted_motion, ref.segment<3>(49), "FootR");

#endif
	return converted_motion;
}

double ReferenceManager::getControlHz() const {
	return mControlHz;
}


}