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
		bool flag_enemy = false;
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
		} else if(ctrl_type == "Punch_enemy") {
			flag_enemy = true;
			newSC = new PUNCH_Controller(ctrl_bvh, ctrl_ppo);	
		} else if(ctrl_type == "Idle_enemy") {
			flag_enemy = true;
			newSC = new IDLE_Controller(ctrl_bvh, ctrl_ppo);	
		}
		if(!flag_enemy) {
			newSC->mParamGoal = newSC->mReferenceManager->GetParamGoal();
			AddSubController(newSC);
		} else {
			newSC->mParamGoal = newSC->mReferenceManager->GetParamGoal();
			mSubControllersEnemy[ctrl_type]= newSC;
		}

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
bool MetaController::IsEnemyPhysics() {
	return mEnemyController.size() > mTargetEnemyIdx && mEnemyController[mTargetEnemyIdx]->mPhysicsMode;
}
void MetaController::Step()
{
	std::pair<Eigen::VectorXd, Eigen::VectorXd> PDTarget = mCurrentController->GetPDTarget(mCharacter);
	Eigen::VectorXd pos = PDTarget.first;
	Eigen::VectorXd vel = PDTarget.second;
	Eigen::VectorXd posEnemy;

	if(IsEnemyPhysics()) {
		// mCurrentEnemyController->GetState(mEnemyController[mTargetEnemyIdx]->mCharacter, 1);
		std::pair<Eigen::VectorXd, Eigen::VectorXd> PDTargetEnemy = mCurrentEnemyController->GetPDTarget(mEnemyController[mTargetEnemyIdx]->mCharacter);
		posEnemy = PDTargetEnemy.first;

	}
	int num_body_nodes = (mCharacter->GetSkeleton()->getNumDofs() - 6)  / 3;

	if(mCurrentController->mCurrentFrameOnPhase <= 5 || mCurrentController->mCurrentFrameOnPhase >= mCurrentController->mReferenceManager->GetPhaseLength()) {
		mCharacter->GetSkeleton()->setVelocities(vel);
		mCharacter->GetSkeleton()->computeForwardKinematics(false,true,false);
	}

	for(int i = 0; i < this->mSimPerCon; i += 2){
		for(int j = 0; j < 2; j++) {
			if(IsEnemyPhysics()) {
				if((mCurrentController->mType=="Punch" && mCurrentController->mCurrentFrameOnPhase >= 16) ||
					(mCurrentController->mType=="Kick" && mCurrentController->mCurrentFrameOnPhase >= 18)) {
					mEnemyController[mTargetEnemyIdx]->mCharacter->GetSkeleton()->setSPDTarget(posEnemy, 300, 16);
				}
				else {
					mEnemyController[mTargetEnemyIdx]->mCharacter->GetSkeleton()->setSPDTarget(posEnemy, 600, 49);
				}
			}
			if(mCurrentController->mType=="Punch" || mCurrentController->mType=="Kick") {
				Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(pos, 600, 49, mWorld->getConstraintSolver());
				for(int j = 0; j < num_body_nodes; j++) {
					int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
					int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
					std::string name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
					double torquelim = mCharacter->GetTorqueLimit(name) * 1.5;
					double torque_norm = torque.block(idx, 0, dof, 1).norm();
				
					torque.block(idx, 0, dof, 1) = std::max(-torquelim, std::min(torquelim, torque_norm)) * torque.block(idx, 0, dof, 1).normalized();
				}

				mCharacter->GetSkeleton()->setForces(torque);
			} else {
				mCharacter->GetSkeleton()->setSPDTarget(pos, 600, 49);

			}

			mWorld->step(false);
		}

	}

	mCurrentController->Step();
	if(IsEnemyPhysics()) {
		mCurrentEnemyController->Step();
	}
	for(int i = 0; i < curEnemyList.size(); i++) {
		mEnemyController[curEnemyList[i]]->Step(mCharacter->GetSkeleton()->getPositions());
	}

	mTargetPositions = mCurrentController->GetCurrentRefPositions();
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	for(int i = 0; i < curEnemyList.size(); i++)
		mRecordEnemyPosition[curEnemyList[i]].push_back(mEnemyController[curEnemyList[i]]->mCharacter->GetSkeleton()->getPositions());
	mTotalSteps += 1;

	if(mIsWaiting && mCurrentController->Synchronizable(mWaiting.first)) {
		std::cout << "make transition to : " << mWaiting.first << " , " << mWaiting.second << std::endl;
		mPrevAction = mCurrentController->mType;
		Eigen::VectorXd prevTargetPos = mCurrentController->GetCurrentRefPositions();

		mCurrentController = mSubControllers[mWaiting.first];
		this->SetAction();
		mCurrentController->Synchronize(mCharacter, prevTargetPos, mWaiting.second);
		mIsWaiting = false;
		std::cout << "make transition done " << std::endl;

	} else if(mCurrentController->mType != "Idle" && mCurrentController->IsEnd()) {
		std::cout << "make transition to : Idle " << std::endl;
		mPrevAction = mCurrentController->mType;
		Eigen::VectorXd prevTargetPos = mCurrentController->GetCurrentRefPositions();

		mCurrentController = mSubControllers["Idle"];
		mCurrentController->Synchronize(mCharacter, prevTargetPos, 0);
	}
	if(IsEnemyPhysics() && mCurrentEnemyController->mType != "Idle_enemy" && mCurrentEnemyController->IsEnd()) {
		std::cout << "make transition to : Idle Enemy" << std::endl;
		mPrevAction = mCurrentEnemyController->mType;
		Eigen::VectorXd prevTargetPos = mEnemyController[mTargetEnemyIdx]->GetPosition();

		mCurrentEnemyController = mSubControllersEnemy["Idle_enemy"];
		mCurrentEnemyController->Synchronize(mEnemyController[mTargetEnemyIdx]->mCharacter, prevTargetPos, 0, 1);
	}
	ClearFallenEnemy();
}
void MetaController::SetAction(){
	mActionSelected = true;
	if(mNextTarget(0) != -10000 && mWaiting.first != "Kick" && mWaiting.first != "Punch") {
		mCurrentController->SetAction(mNextTarget);
		return;
	}
	Eigen::VectorXd action;
	if(mWaiting.first == "Kick"){
		action.resize(3);
		action.setZero();
		
		Eigen::Vector6d root =  mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		Eigen::Vector3d root_ori = root.segment<3>(0);
		root_ori = projectToXZ(root_ori);
		Eigen::AngleAxisd root_aa(root_ori.norm(), root_ori.normalized());
			
		Eigen::Vector3d hand = mEnemyController[mTargetEnemyIdx]->GetCOM();
		hand = hand - root.segment<3>(3);
		hand(1) = 0;
		Eigen::Vector3d dir = root_aa.inverse() * hand;
		double norm = dir.norm();
		dir.normalize();
		if(mNextTarget(0) != -10000)
			action <<  dir(2), 1.35, mNextTarget(0);
		else
			action <<  dir(2), 1.35, 1.0;
	
	} else if(mWaiting.first == "Punch"){
		action.resize(4);

		Eigen::Vector6d root =  mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		Eigen::Vector3d root_ori = root.segment<3>(0);
		root_ori = projectToXZ(root_ori);
		Eigen::AngleAxisd root_aa(root_ori.norm(), root_ori.normalized());
			
		Eigen::Vector3d hand = mEnemyController[mTargetEnemyIdx]->GetCOM();
		hand = hand - root.segment<3>(3);
		hand(1) = 0;
		Eigen::Vector3d dir = root_aa.inverse() * hand;
		double norm = dir.norm();
		dir.normalize();
		if(mNextTarget(0) != -10000)
			action << dir(2), 1.2, norm, mNextTarget(0);
		else
			action << dir(2), 1.2, norm, 1.4;

		// this->mWorld->addSkeleton(mEnemyController[mTargetEnemyIdx]->mCharacter);
		// mEnemyController[i]->SetPhysicsMode(true);
	
	} else if(mWaiting.first == "Pivot"){
		action.resize(1);
		
		Eigen::Vector6d root = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		Eigen::Vector3d root_ori = mCharacter->GetSkeleton()->getPositions().segment<3>(0);
		root_ori = projectToXZ(root_ori);
		Eigen::AngleAxisd root_aa(root_ori.norm(), root_ori.normalized());
		double angle = 0;
		double distance = 0;
		int idx = 0;
		
		Eigen::Vector3d point_local = mEnemyController[mTargetEnemyIdx]->GetCOM() - root.segment<3>(3);
		point_local(1) = 0;
		point_local = root_aa.inverse() * point_local; 
		double angle_cur = -atan2(point_local(0), point_local(2));

		if(angle_cur < 0)
			angle_cur = 2 * M_PI + angle_cur; 

		action(0) = (angle_cur + 0.1 * M_PI) / 1.4;
	} else if(mWaiting.first == "Dodge"){
		action.resize(1);

		Eigen::Vector6d root = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
		Eigen::Vector3d root_ori = mCharacter->GetSkeleton()->getPositions().segment<3>(0);
		root_ori = projectToXZ(root_ori);
		Eigen::AngleAxisd root_aa(root_ori.norm(), root_ori.normalized());
		double angle = 0;
		double distance = 0;
		int idx = 0;
		
		Eigen::Vector3d point_local = mEnemyController[mTargetEnemyIdx]->GetCOM() - root.segment<3>(3);
		point_local(1) = 0;
		point_local = root_aa.inverse() * point_local; 
		double angle_cur = atan2(point_local(0), point_local(2));

		action(0) = angle_cur;
	}

	mCurrentController->SetAction(action);

}
double MetaController::GetCurrentPhase(){
	if(!mCurrentController) 
		return 0;

	return mCurrentController->mCurrentFrameOnPhase;
}

void MetaController::SwitchController(std::string type, Eigen::VectorXd target, int frame, bool isEnemy)
{
	mNextTarget = target;
	if(!isEnemy) {
		mIsWaiting = true;
		mActionSelected = false;
		mWaiting = std::pair<std::string, double>(type, frame);
		std::cout << "waiting: " << type << " , " << frame << std::endl;
	} else {
		Eigen::VectorXd prevTargetPos = mEnemyController[mTargetEnemyIdx]->GetPosition();

		mCurrentEnemyController = mSubControllersEnemy[type];
		mCurrentEnemyController->Synchronize(mEnemyController[mTargetEnemyIdx]->mCharacter, prevTargetPos, 0);
	}
}
std::string MetaController::GetNextAction()
{
	if(mIsWaiting)
		return mWaiting.first;
	else
		return mCurrentController->mType;
}
int 
MetaController::
AddNewEnemy(Eigen::VectorXd d) {

	int i = mEnemyController.size();
	Eigen::Vector3d p_ch = mCharacter->GetSkeleton()->getPositions().segment<3>(3);
	Eigen::Vector3d d_ch = mCharacter->GetSkeleton()->getPositions().segment<3>(0);

	Eigen::Vector3d p_em;
	Eigen::AngleAxisd aa = Eigen::AngleAxisd(d_ch.norm(), d_ch.normalized());
	if(d(0) >= 0) {
		Eigen::Vector3d dir = Eigen::Vector3d(sin(d(0)*M_PI), 0, cos(d(0)*M_PI));
		dir = aa * dir*d(1);
		p_em += dir;
	}
	else {
		d(0) = mUniform(mMT) * 2;
		Eigen::Vector3d dir = Eigen::Vector3d(sin(d(0)*M_PI), 0, cos(d(0)*M_PI));
		dir = aa * dir*1.3;
		p_em += dir;
	}
	mEnemyController.push_back(new EnemyKinController(p_em, p_ch));

	std::vector<Eigen::VectorXd> record;
	record.clear();
	mRecordEnemyPosition.push_back(record);
	mRecordEnemyTiming.push_back(mRecordPosition.size());
	curEnemyList.push_back(i);

	return i;
}
void
MetaController::
ClearFallenEnemy() {
	std::vector<int> newlist;
	for(int i = 0; i < curEnemyList.size(); i++) {
		if(mEnemyController[curEnemyList[i]]->GetPosition()(4) < 0.15) {
			std::cout << "clear enemy : " << curEnemyList[i] << std::endl;
		} else {
			newlist.push_back(curEnemyList[i]);
		}
	}
	curEnemyList = newlist;
}
void
MetaController::
ToggleTargetPhysicsMode() {
	bool prevMode = mEnemyController[mTargetEnemyIdx]->mPhysicsMode;
	if(prevMode == false) {
		auto& skel = mEnemyController[mTargetEnemyIdx]->mCharacter->GetSkeleton();
		skel->clearConstraintImpulses();
		skel->clearInternalForces();
		skel->clearExternalForces();

		Eigen::VectorXd prevTargetPos = mEnemyController[mTargetEnemyIdx]->GetPosition();
		mCurrentEnemyController = mSubControllersEnemy["Idle_enemy"];
		mCurrentEnemyController->Synchronize(mEnemyController[mTargetEnemyIdx]->mCharacter, prevTargetPos, 0);

		mWorld->addSkeleton(mEnemyController[mTargetEnemyIdx]->mCharacter->GetSkeleton());
	} else {
		mWorld->removeSkeleton(mEnemyController[mTargetEnemyIdx]->mCharacter->GetSkeleton());
		mEnemyController[mTargetEnemyIdx]->SetAction("box_idle");

	}
	mEnemyController[mTargetEnemyIdx]->mPhysicsMode = !prevMode;
}
void
MetaController::
SwitchMainTarget() {
	mEnemyController[mTargetEnemyIdx]->mPhysicsMode = false;
	mTargetEnemyIdx += 1;
}
Eigen::VectorXd
MetaController::
GetEnemyPositions(int i) {
	return mEnemyController[i]->GetPosition();
}
void
MetaController::
SaveAsBVH(std::string filename, std::vector<Eigen::VectorXd> record) {
	std::string path = std::string(CAR_DIR)+ std::string("/build/interactive/")+filename;
	std::cout << "save results to" << path << std::endl;
	std::vector<std::string> HIERARCHY = mCurrentController->mReferenceManager->GetHierarchyStr();

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
	
	ofs << "HIERARCHY" << std::endl;
	ofs << "ROOT Hips" << std::endl;
	for(int i = 0; i < HIERARCHY.size(); i++) {
		ofs << HIERARCHY[i] << std::endl;
	}
	ofs << "MOTION" << std::endl;
	ofs << "Frames: " << std::to_string(record.size()) << std::endl;
	ofs << "Frame Time:	0.0333333" << std::endl;
	
	for(auto t: record) {
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
	std::cout << "saved position: " << record.size() << std::endl;
	ofs.close();
}
void
MetaController::
SaveAll(std::string filename) {
	SaveAsBVH(filename+std::string("_main"), mRecordPosition);
	for(int i = 0; i < mEnemyController.size(); i++) {
		SaveAsBVH(filename+std::string("_enemy")+std::to_string(i), mRecordEnemyPosition[i]);
	}

	//save timing
	std::string path = std::string(CAR_DIR)+ std::string("/build/interactive/")+filename+std::string("_enemyTiming");
	std::ofstream ofs(path);

	for(int i = 0; i < mRecordEnemyTiming.size(); i++) {
		ofs << i << " "<< mRecordEnemyTiming[i] << std::endl;
	}
	ofs.close();

}
} //end of namespace DPhy
