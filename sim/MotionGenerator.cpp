#include "MotionGenerator.h"
#include "Functions.h"
#include <cstdlib>
#include <ctime>
#include <fstream>

namespace DPhy
{
MotionGenerator::
MotionGenerator(Character* character)
: mCharacter(character), mCurrentMotion(0), mNextMotion(0), mMotionStartTime(0), mMotionDuration(0)
{
	this->mCOMOffsetGlobal.setZero();
	this->mOrientationOffsetGlobal.setIdentity();

	this->mPositionDifferenceWithLastMotion.resize(this->mCharacter->GetSkeleton()->getNumDofs());
	srand(time(NULL));
}

double
MotionGenerator::
getMaxTime(){
	return this->mBVHs[this->mCurrentMotion]->GetMaxTime();
}

void
MotionGenerator::
Initialize()
{
	assert(mBVHs.size() > 0);

	this->mCurrentMotion = this->mNextMotion = rand()%(this->mBVHs.size());
	this->mMotionStartTime = 0.0;
	this->mMotionDuration = this->mBVHs[this->mCurrentMotion]->GetMaxTime();

	this->mFirstPositions = this->mCharacter->GetTargetPositions(this->mBVHs[this->mCurrentMotion], 0);

	this->mCOMOffsetGlobal.setZero();
	this->mCOMOffsetCurrent = this->mFirstPositions.segment<3>(3);
	this->mCOMOffsetCurrent[1] = 0;

	this->mOrientationOffsetGlobal.setIdentity();
	this->mOrientationOffsetCurrent = GetYRotation(DARTPositionToQuaternion(this->mFirstPositions.segment<3>(0)));

	this->mPositionDifferenceWithLastMotion.setZero();
}

void printQuaternion(Eigen::Quaterniond q){
	std::cout << q.w() << ", " << q.vec().transpose() << std::endl;
}

Eigen::VectorXd 
MotionGenerator::
applyOffset(Eigen::VectorXd p, double time_in_motion){
	double transition_time = 1.0;
	transition_time = std::min(transition_time, this->mMotionDuration);
	if(time_in_motion < transition_time){
		double t = dart::math::clip(1.0-time_in_motion / transition_time, 0.0, 1.0);
		// root joint
		Eigen::Quaterniond root_cur = DARTPositionToQuaternion(p.segment<3>(0))
					* DARTPositionToQuaternion(t*(this->mPositionDifferenceWithLastMotion.segment<3>(0)));

		p.segment<3>(0) = QuaternionToDARTPosition(root_cur);

		// else
		for(int i = 0; i < this->mCharacter->GetSkeleton()->getNumJoints(); i++){
			dart::dynamics::Joint* jn = this->mCharacter->GetSkeleton()->getJoint(i);			
			if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr){
				int index = jn->getIndexInSkeleton(0);
				Eigen::Quaterniond first_pos = DARTPositionToQuaternion(this->mFirstPositions.segment<3>(index));
				Eigen::Quaterniond cur_diff = first_pos.inverse()
										* DARTPositionToQuaternion(p.segment<3>(index));

				Eigen::Quaterniond cur = DARTPositionToQuaternion(p.segment<3>(index))
											* DARTPositionToQuaternion(t*(this->mPositionDifferenceWithLastMotion.segment<3>(index)));
				p.segment<3>(index) = QuaternionToDARTPosition(cur);
			}
		}
	}

	Eigen::Vector3d tp = p.segment<3>(3);
	tp[1] = 0;
	tp = this->mCOMOffsetGlobal + (this->mOrientationOffsetGlobal* this->mOrientationOffsetCurrent.inverse())._transformVector(tp-this->mCOMOffsetCurrent);
	p[3] = tp[0];
	p[5] = tp[2];
	p.segment<3>(0) = QuaternionToDARTPosition(this->mOrientationOffsetGlobal * this->mOrientationOffsetCurrent.inverse() * DARTPositionToQuaternion(p.segment<3>(0)));

	return p;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
MotionGenerator::
getMotion(double time)
{
	assert(time >= this->mMotionStartTime);

	if(time >= this->mMotionStartTime + this->mMotionDuration){
		// change motion to next motion
		Eigen::VectorXd lastPositions = this->mCharacter->GetTargetPositions(this->mBVHs[this->mCurrentMotion], this->mMotionDuration-this->mBVHs[this->mCurrentMotion]->GetTimeStep());
		lastPositions = this->applyOffset(lastPositions, this->mMotionDuration-this->mBVHs[this->mCurrentMotion]->GetTimeStep());

		this->mCurrentMotion = this->mNextMotion = rand()%(this->mBVHs.size());
		this->mMotionStartTime += this->mMotionDuration;
		this->mMotionDuration = this->mBVHs[this->mCurrentMotion]->GetMaxTime();

		this->mFirstPositions = this->mCharacter->GetTargetPositions(this->mBVHs[this->mCurrentMotion], 0);

		this->mCOMOffsetGlobal[0] = lastPositions[3];
		this->mCOMOffsetGlobal[1] = 0;
		this->mCOMOffsetGlobal[2] = lastPositions[5];

		this->mCOMOffsetCurrent = this->mFirstPositions.segment<3>(3);
		this->mCOMOffsetCurrent[1] = 0;

		this->mOrientationOffsetGlobal = GetYRotation(DARTPositionToQuaternion(lastPositions.segment<3>(0)));
		this->mOrientationOffsetCurrent = GetYRotation(DARTPositionToQuaternion(this->mFirstPositions.segment<3>(0)));


		this->mFirstPositions = this->applyOffset(this->mFirstPositions, 10.0); // 10.0 for apply offset to root joint only

		this->mPositionDifferenceWithLastMotion.setZero();
		this->mPositionDifferenceWithLastMotion.segment<3>(0) 
			= QuaternionToDARTPosition(
				DARTPositionToQuaternion(this->mFirstPositions.segment<3>(0)).inverse()
				* DARTPositionToQuaternion(lastPositions.segment<3>(0))
			);

		// root joint
		Eigen::Quaterniond last_q = DARTPositionToQuaternion(lastPositions.segment<3>(0));
		last_q = GetYRotation(last_q).inverse() * last_q;

		Eigen::Quaterniond first_q = DARTPositionToQuaternion(this->mFirstPositions.segment<3>(0));
		first_q = GetYRotation(first_q).inverse() * first_q;

		this->mPositionDifferenceWithLastMotion.segment<3>(0) = QuaternionToDARTPosition(first_q.inverse()*last_q);
		// else
		for(int i = 0; i < this->mCharacter->GetSkeleton()->getNumJoints(); i++){
			dart::dynamics::Joint* jn = this->mCharacter->GetSkeleton()->getJoint(i);			
			if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr){
				int index = jn->getIndexInSkeleton(0);
				this->mPositionDifferenceWithLastMotion.segment<3>(index)
				 = QuaternionToDARTPosition(
				 	DARTPositionToQuaternion(this->mFirstPositions.segment<3>(index)).inverse()
				    * DARTPositionToQuaternion(lastPositions.segment<3>(index))
				   );
			}
		}
	}
	double time_in_motion = time - this->mMotionStartTime;
	assert(time_in_motion < this->mMotionDuration);

	Eigen::VectorXd p = this->mCharacter->GetTargetPositions(this->mBVHs[this->mCurrentMotion], time_in_motion);
	p = this->applyOffset(p, time_in_motion);

	double p_time = time_in_motion - 0.05;
	if(time_in_motion <= 0.05)
		p_time = time_in_motion + 0.05;

	Eigen::VectorXd p1 = this->mCharacter->GetTargetPositions(this->mBVHs[this->mCurrentMotion], p_time);
	p1 = this->applyOffset(p1, p_time);

	Eigen::VectorXd v;
	if(time_in_motion <= 0.05)
		v = this->mCharacter->GetSkeleton()->getPositionDifferences(p1, p)*20;
	else
		v = this->mCharacter->GetSkeleton()->getPositionDifferences(p, p1)*20;

	// std::cout << std::endl;
	// std::cout << "Velocity debug" << std::endl;
	// std::cout << time_in_motion << ", " << p_time << std::endl;
	// std::cout << p.segment<3>(3).transpose() << std::endl;
	// std::cout << p1.segment<3>(3).transpose() << std::endl;
	// std::cout << ((p1.segment<3>(3) - p.segment<3>(3))*20).transpose() << std::endl;
	// std::cout << v.segment<3>(3).transpose() << std::endl;

	return std::make_pair(p, v);
}


void 
MotionGenerator::
setNext(int index)
{
	assert(index < this->mBVHs.size());
	assert(index >= 0);

	this->mNextMotion = index;
}

void 
MotionGenerator::
addBVH(std::string motionfilename)
{
	BVH* bvh = new BVH();
	bvh->Parse(motionfilename);
	this->mCharacter->InitializeBVH(bvh);

	this->mBVHs.push_back(bvh);

}

void 
MotionGenerator::
addBVHs(std::string motionfilename)
{
	std::ifstream input(motionfilename);
	std::vector<std::string> motionsfiles;
	std::string line;
	while(std::getline(input, line)){
		if(line == "end") break;
		std::cout << line << std::endl;
		motionsfiles.push_back(line);
	}
	for(auto filename : motionsfiles){
		this->addBVH(std::string(DPHY_DIR)+filename);
	}
	input.close();
}

}