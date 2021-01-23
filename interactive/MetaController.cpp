#include "MetaController.h"
#include <tinyxml.h>

namespace DPhy
{	

MetaController::MetaController()
: mControlHz(30),mSimulationHz(150), mRD(), mMT(mRD()), mUniform(0.0, 1.0) 
{
	this->mSimPerCon = mSimulationHz / mControlHz;
	this->mWorld = std::make_shared<dart::simulation::World>();

	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81, 0));

	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml")).first;
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());

	Eigen::VectorXd kp(this->mCharacter->GetSkeleton()->getNumDofs()), kv(this->mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();
	kv.setZero();
	this->mCharacter->SetPDParameters(kp,kv);

	LoadControllers();
	mCurrentController = mSubControllers["Idle"];
	mPrevController = nullptr;
}
void MetaController::LoadControllers()
{
	std::string ctrl_path = std::string(CAR_DIR)+ std::string("/scene/s1_ctrl.xml");
	std::cout<<"loadControllers: "<<ctrl_path<<std::endl;
	TiXmlDocument doc;
	if(!doc.LoadFile(ctrl_path)){
		std::cout << "Can't open scene file : " << ctrl_path << std::endl;
	}

	TiXmlElement *skeldoc = doc.FirstChildElement("ControllerList");
	
	for(TiXmlElement *body = skeldoc->FirstChildElement("SubController"); body != nullptr; body = body->NextSiblingElement("SubController")){
		
		std::string ctrl_type = body->Attribute("type");
		std::string ctrl_bvh = body->Attribute("bvh");
		std::string ctrl_ppo = body->Attribute("ppo");

		std::cout<< "================ ADD SUB Controller: "<<ctrl_type<<" :: "<<ctrl_bvh<<" , "<<ctrl_ppo<<std::endl;
		SubController* newSC;
		 if(ctrl_type == "Punch"){
			newSC = new PUNCH_Controller(ctrl_bvh, ctrl_ppo);
		}else if(ctrl_type == "Kick"){
			newSC = new KICK_Controller(ctrl_bvh, ctrl_ppo);
		}else if(ctrl_type == "Idle"){
			newSC = new IDLE_Controller(ctrl_bvh, ctrl_ppo);	
		}else if(ctrl_type == "Dodge"){
			newSC = new DODGE_Controller(ctrl_bvh, ctrl_ppo);	
		} else if(ctrl_type == "Pivot"){
			newSC = new PIVOT_Controller(ctrl_bvh, ctrl_ppo);	
		}else{
			std::cout<<" NOT A PROPER COTNROLLER TYPE : "<<ctrl_type<<std::endl;
			continue;
		}

		newSC->mParamGoal = newSC->mReferenceManager->GetParamGoal();
		AddSubController(newSC);
	}
}
void MetaController::Reset()
{
	this->mWorld->reset();
	auto& skel = mCharacter->GetSkeleton();
	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();
	mRecordPosition.clear();

	this->mTotalSteps = 0; 
	mCurrentController = mSubControllers["Idle"];
	Eigen::VectorXd prevTargetPos(skel->getNumDofs());
	prevTargetPos.setZero();

	mCurrentController->Synchronize(mCharacter, prevTargetPos,0);

	mTargetPositions = mCurrentController->GetCurrentRefPositions();
	Eigen::VectorXd vel(mTargetPositions.rows());
	vel.setZero();
	
	skel->setPositions(mTargetPositions);
	skel->setVelocities(vel);
	skel->computeForwardKinematics(true,true,false);

	mIsWaiting = false;

}
void MetaController::Step()
{
	mCurrentController->Step(mWorld, mCharacter);
	mTargetPositions = mCurrentController->GetCurrentRefPositions();
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mTotalSteps += 1;
	if(mIsWaiting && mCurrentController->Synchronizable(mWaiting.first)) {
		std::cout << "make transition to : " << mWaiting.first << " , " << mWaiting.second << std::endl;
		Eigen::VectorXd prevTargetPos = mCurrentController->GetCurrentRefPositions();

		mCurrentController = mSubControllers[mWaiting.first];
		this->SetAction();
		mCurrentController->Synchronize(mCharacter, prevTargetPos, mWaiting.second);
		mIsWaiting = false;
		std::cout << "make transition done " << std::endl;

	} else if(mCurrentController->mType != "Idle" && mCurrentController->IsEnd()) {
		std::cout << "make transition to : Idle " << std::endl;
		Eigen::VectorXd prevTargetPos = mCurrentController->GetCurrentRefPositions();

		mCurrentController = mSubControllers["Idle"];
		mCurrentController->Synchronize(mCharacter, prevTargetPos, 22);
	}
}
void MetaController::SetAction(){
	mActionSelected = true;
	Eigen::VectorXd action;
	if(mWaiting.first == "Kick"){
		action.resize(2);
		action.setZero();
		if(mHitPoints.size() != 0) {
			Eigen::Vector6d root = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
			Eigen::Vector3d root_ori = mCharacter->GetSkeleton()->getPositions().segment<3>(0);
			root_ori = projectToXZ(root_ori);
			Eigen::AngleAxisd root_aa(root_ori.norm(), root_ori.normalized());
			double angle = 0;
			double distance = 0;
			int idx = 0;
			for(int i = 0; i < mHitPoints.size(); i++) {
				Eigen::Vector3d point_local = mHitPoints[i] - root.segment<3>(3);
				point_local(1) = 0;
				point_local = root_aa.inverse() * point_local; 
				double angle_cur = atan2(point_local(2), point_local(0));

				angle_cur = 0.5 * M_PI - angle_cur;
				if(angle_cur < 0)
					angle_cur += 2 * M_PI;

				if(i == 0) {
					angle = angle_cur;
					distance = point_local.norm();
				} else if(distance > 1.5 && point_local.norm() <= 1.5) {
					angle = angle_cur;
					distance = point_local.norm();
					idx = i;
				} else if( angle > angle_cur && point_local.norm() <= 1.5) {
					angle = angle_cur;
					distance = point_local.norm();
					idx = i;

				}
			}
			action(0) = angle / 6.0 * 2;
			action(1) = mHitPoints[idx](1) - 0.2;
		}
	} else if(mWaiting.first == "Punch"){
		action.resize(4);
		action.setZero();
		if(mHitPoints.size() != 0) {
			Eigen::Vector6d root = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
			Eigen::Vector3d root_ori = root.segment<3>(0);
			root_ori = projectToXZ(root_ori);
			Eigen::AngleAxisd root_aa(root_ori.norm(), root_ori.normalized());
			double angle = 0;
			double distance = 0;
			int idx = 0;
			for(int i = 0; i < mHitPoints.size(); i++) {
				Eigen::Vector3d point_local = mHitPoints[i] - root.segment<3>(3);
				point_local(1) = 0;
				point_local = root_aa.inverse() * point_local; 
				double angle_cur = atan2(point_local(2), point_local(0));

				if(angle_cur < 0.5 * M_PI)
					angle_cur += 2* M_PI; 
				angle_cur -= 0.5 * M_PI;

				if(i == 0) {
					angle = angle_cur;
					distance = point_local.norm();
				} else if(distance > 1.5 && point_local.norm() <= 1.5) {
					angle = angle_cur;
					distance = point_local.norm();
					idx = i;
				} else if(angle > angle_cur && point_local.norm() <= 1.5) {
					angle = angle_cur;
					distance = point_local.norm();
					idx = i;
				}
			}

			Eigen::Vector3d hand = mHitPoints[idx];
			hand = hand - root.segment<3>(3);
			hand(1) = 0;
			Eigen::Vector3d dir = root_aa.inverse() * hand;
			double norm = dir.norm();
			dir.normalize();
			action << dir(2), mHitPoints[idx](1), norm, 0.4;
		}
	} else if(mWaiting.first == "Pivot"){
		action.resize(1);
		action(0) = 0.0;
		if(mHitPoints.size() != 0) {
			Eigen::Vector6d root = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
			Eigen::Vector3d root_ori = mCharacter->GetSkeleton()->getPositions().segment<3>(0);
			root_ori = projectToXZ(root_ori);
			Eigen::AngleAxisd root_aa(root_ori.norm(), root_ori.normalized());
			double angle = 0;
			double distance = 0;
			int idx = 0;
			for(int i = 0; i < mHitPoints.size(); i++) {
				Eigen::Vector3d point_local = mHitPoints[i] - root.segment<3>(3);
				point_local(1) = 0;
				point_local = root_aa.inverse() * point_local; 
				double angle_cur = atan2(point_local(2), point_local(0));

				if(angle_cur < 0.5 * M_PI)
					angle_cur += 2* M_PI; 
				angle_cur -= 0.5 * M_PI;

				if(i == 0) {
					angle = angle_cur;
					distance = point_local.norm();
				} else if(distance > 1.5 && point_local.norm() <= 1.5) {
					angle = angle_cur;
					distance = point_local.norm();
					idx = i;
				} else if(angle > angle_cur && point_local.norm() <= 1.5) {
					angle = angle_cur;
					distance = point_local.norm();
					idx = i;
				}
			}
			if(angle > 0.1 * M_PI) {
				action(0) = std::min(0.3 * (angle - 0.1 * M_PI), 0.7);
			}
		}
	} else if(mWaiting.first == "Dodge"){
		action.resize(1);
		action(0) = mUniform(mMT);
	}

	mCurrentController->SetAction(action);

}
void MetaController::SwitchController(std::string type, int frame)
{
	mIsWaiting = true;
	mActionSelected = false;
	mWaiting = std::pair<std::string, double>(type, frame);
	std::cout << "waiting: " << type << " , " << frame << std::endl;
}
std::string MetaController::GetNextAction()
{
	if(mIsWaiting)
		return mWaiting.first;
	else
		return mCurrentController->mType;
}
void 
MetaController::
AddNewRandomHitPoint() {
	Eigen::Vector3d p = GetCOM();	
	double distance = mUniform(mMT);
	double dir = 2 * (mUniform(mMT) - 0.5);
	Eigen::Vector3d dir3d;
	if(mUniform(mMT) > 0.5) {
		dir3d = Eigen::Vector3d(dir, 0, sqrt(1-dir*dir));
	} else {
		dir3d = Eigen::Vector3d(dir, 0, -sqrt(1-dir*dir));
	}
	p += (distance * 0.5 + 0.5) * dir3d;
	p(1) = 0.5 + mUniform(mMT);
	mHitPoints.push_back(p);
}
void
MetaController::
SaveAsBVH(std::string filename) {
	std::string path = std::string(CAR_DIR)+ std::string("/build/interactive/")+filename;
	std::cout << "save results to" << path << std::endl;

	std::ofstream ofs(path);
	std::vector<std::string> bvh_order;
	bvh_order.push_back("Hips");
	bvh_order.push_back("Spine");
	bvh_order.push_back("Spine1");
	bvh_order.push_back("Spine2");
	bvh_order.push_back("Neck");
	bvh_order.push_back("Head");
	bvh_order.push_back("LeftShoulder");
	bvh_order.push_back("LeftArm");
	bvh_order.push_back("LeftForeArm");
	bvh_order.push_back("LeftHand");
	bvh_order.push_back("RightShoulder");
	bvh_order.push_back("RightArm");
	bvh_order.push_back("RightForeArm");
	bvh_order.push_back("RightHand");
	bvh_order.push_back("LeftUpLeg");
	bvh_order.push_back("LeftLeg");
	bvh_order.push_back("LeftFoot");
	bvh_order.push_back("LeftToe");
	bvh_order.push_back("RightUpLeg");
	bvh_order.push_back("RightLeg");
	bvh_order.push_back("RightFoot");
	bvh_order.push_back("RightToe");
	
	ofs << "MOTION" << std::endl;
	ofs << "Frames: " << std::to_string(mRecordPosition.size()) << std::endl;
	ofs << "Frame Time:	0.0333333" << std::endl;
	
	for(auto t: mRecordPosition) {
		ofs << t.segment<3>(3).transpose() * 100 << " ";

		for(int i = 0; i < bvh_order.size(); i++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(bvh_order[i])->getParentJoint()->getIndexInSkeleton(0);
			Eigen::AngleAxisd aa(t.segment<3>(idx).norm(), t.segment<3>(idx).normalized());
			Eigen::Matrix3d m;
			m = aa;
			Eigen::Vector3d v = dart::math::matrixToEulerZXY(m);
			ofs << v.transpose() * 180 / M_PI << " ";			
		}
		ofs << std::endl;
	}
	std::cout << "saved position: " << mRecordPosition.size() << std::endl;
	ofs.close();
}
} //end of namespace DPhy
