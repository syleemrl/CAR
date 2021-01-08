#include "MetaController.h"
#include <tinyxml.h>

namespace DPhy
{	

MetaController::MetaController(std::string ctrl, std::string scene_obj, std::string scenario)
: mControlHz(30),mSimulationHz(150),mCurrentFrame(0), mCurrentFrameOnPhase(0),terminationReason(-1), mIsTerminal(false)
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

	if(scene_obj!="") loadSceneObjects(std::string(CAR_DIR)+std::string("/scene/") + scene_obj + std::string(".xml"));

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());

	Eigen::VectorXd kp(this->mCharacter->GetSkeleton()->getNumDofs()), kv(this->mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();
	kv.setZero();
	this->mCharacter->SetPDParameters(kp,kv);

	mInterestedDof = mCharacter->GetSkeleton()->getNumDofs() - 6;
	// mRewardDof = mCharacter->GetSkeleton()->getNumDofs();

	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 
	
	mActions = Eigen::VectorXd::Zero(mInterestedDof + 1);
	mActions.setZero();

	mEndEffectors.clear();
	mEndEffectors.push_back("RightFoot");
	mEndEffectors.push_back("LeftFoot");
	mEndEffectors.push_back("LeftHand");
	mEndEffectors.push_back("RightHand");
	mEndEffectors.push_back("Head");

	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	this->mPDTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mPDTargetVelocities = Eigen::VectorXd::Zero(dof);

	// this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();

	mTimeElapsed = 0;


	// load SubControllers
	assert(ctrl!= "");
	loadControllers(std::string(CAR_DIR)+ std::string("/scene/")+ctrl+std::string(".xml"));

	if(scenario != ""){
		//TODO
	}else{
		mCurrentController= mSubControllers["FW_JUMP"];
	}
	mPrevController = nullptr;

	mSubControllers["WALL_JUMP"]->mParamGoal= mSubControllers["WALL_JUMP"]->mReferenceManager->GetParamGoal();
	mSubControllers["FW_JUMP"]->mParamGoal= mSubControllers["FW_JUMP"]->mReferenceManager->GetParamGoal();


	mRef1 = mCurrentController->mReferenceManager;
	mTime1 = 0;
	mAlign1 = Eigen::Isometry3d::Identity();

	mRef2 = nullptr;
	mTime2 = 0;
	mAlign2 = Eigen::Isometry3d::Identity();

	runScenario();
}

void MetaController::reset()
{
	this->mWorld->reset();
	auto& skel = mCharacter->GetSkeleton();
	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();

	bool isAdaptive = true;
	this->mCurrentFrame = 0; 
	this->mCurrentFrameOnPhase = 0;
	this->mTime1 = mCurrentFrame;

	// this->mStartFrame = this->mCurrentFrame;
	// this->nTotalSteps = 0;
	this->mTimeElapsed = 0;

	Motion* p_v_target;
	p_v_target = GetMotion(0, true);

	// std::cout<<p_v_target->GetPosition().transpose()<<std::endl<<std::endl;
	// std::cout<<p_v_target->GetVelocity().transpose()<<std::endl;


	// Eigen::VectorXd new_p = p_v_target->GetPosition();
	// new_p.tail<40>().setZero();

	// Eigen::VectorXd new_v = p_v_target->GetPosition();
	// new_v.tail<40>().setZero();

	// p_v_target->SetPosition(new_p);
	// p_v_target->SetVelocity(new_v);

	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	this->mPDTargetPositions = mTargetPositions;
	this->mPDTargetVelocities = mTargetVelocities;

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	skel->computeForwardKinematics(true,true,false);

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;
	
	// ClearRecord();
	SaveStepInfo();

	// mRootZero = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	// mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	mPrevTargetPositions = mTargetPositions;
	
	mPrevFrame = mCurrentFrame;
	mPrevFrame2 = mPrevFrame;
	
	mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	mTimeQueue.push(mCurrentFrame);
	mAdaptiveStep = 1;

	mTiming= std::vector<double>();
	mTiming.push_back(mCurrentFrame);

	mCurrentController->reset();

}
void MetaController::runScenario(){
	//TODO
	std::cout<<"mCurrent Controller Type : "<<mCurrentController->mType<<std::endl;

	// mCurrentController= mSubControllers["FW_JUMP"];
	// mRef1 = mCurrentController->mReferenceManager;
	// mAlign1 = Eigen::Isometry3d::Identity();

	// mAlign1.linear()<<  -0.978497,  6.93889e-18 , 0.206261,
	// 		-1.38778e-17,           1  ,     0,
	// 	   -0.206261,   2.1684e-17,    -0.978497;
 //    mAlign1.translation() = Eigen::Vector3d(1.11078, -0.00937659 , 2.19823);

	this->reset();

	while(! IsTerminalState()){

		std::cout<<"\n@ "<<mTime1;
		if(mRef2!=nullptr) std::cout<<" / "<<mTime2<<" / "<<((double)mBlendStep/2/(mBlendMargin+1));
		std::cout<<std::endl;
		Eigen::VectorXd state = GetState();

		p::object a = this->mCurrentController->mPPO.attr("run")(DPhy::toNumPyArray(state));
		np::ndarray na = np::from_object(a);
		Eigen::VectorXd action = DPhy::toEigenVector(na,this->GetNumAction());
		this->SetAction(action);
		this->Step();	
		
		// if(mCurrentFrame >=120){
		// 	scenario_done = true;
		// 	break;
		// }	

		int blendFrame1 = 40; //66; //71;
		int blendFrame2 = 4; //0; //63;

		if(control_mode == 0 && mTime1 >= blendFrame1){

			Eigen::Isometry3d prev_cycle_end= mSubControllers["FW_JUMP"]->mReferenceManager->GetRootTransform(blendFrame1, true);
			Eigen::Isometry3d cycle_start= mSubControllers["WALL_JUMP"]->mReferenceManager->GetRootTransform(blendFrame2, true);

			mAlign1 = prev_cycle_end*cycle_start.inverse();
			Eigen::Vector3d p01 = dart::math::logMap(mAlign1.linear());			
			mAlign1.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));

			Eigen::Isometry3d cycle_start_edit = cycle_start;
			cycle_start_edit.linear() = mAlign1.linear().inverse()*prev_cycle_end.linear();
			mAlign1 = prev_cycle_end*cycle_start_edit.inverse();

			control_mode = 1;
			mCurrentController= mSubControllers["WALL_JUMP"];
			mRef1 = mSubControllers["WALL_JUMP"]->mReferenceManager;
			mTime1 = blendFrame2;
			mCurrentFrameOnPhase = mTime1;
			mCurrentController->setCurObject(mSceneObjects[0]);
			mCurrentController->reset(mCurrentFrame, mCurrentFrameOnPhase);

			auto& obj = mCurrentController->mCurObject;
			Eigen::VectorXd p = obj->getPositions();
			MultiplyRootTransform(p, mAlign1, false);
			p[5]+=0.1;
			obj->setPositions(p);
		}

	// 	if(control_mode == 0 && mTime1 >= blendFrame1){

	// 		Eigen::Isometry3d prev_cycle_end= mSubControllers["WALL_JUMP"]->mReferenceManager->GetRootTransform(blendFrame1, true);
	// 		Eigen::Isometry3d cycle_start= mSubControllers["FW_JUMP"]->mReferenceManager->GetRootTransform(blendFrame2, true);

	// 		mAlign1 = prev_cycle_end*cycle_start.inverse();
	// 		Eigen::Vector3d p01 = dart::math::logMap(mAlign1.linear());			
	// 		mAlign1.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));

	// 		Eigen::Isometry3d cycle_start_edit = cycle_start;
	// 		cycle_start_edit.linear() = mAlign1.linear().inverse()*prev_cycle_end.linear();
	// 		mAlign1 = prev_cycle_end*cycle_start_edit.inverse();

	// 		control_mode = 1;
	// 		mCurrentController= mSubControllers["FW_JUMP"];
	// 		mRef1 = mSubControllers["FW_JUMP"]->mReferenceManager;
	// 		mTime1 = blendFrame2;
	// 		mCurrentFrameOnPhase = mTime1;

	// // auto& skel = mCharacter->GetSkeleton();

	// // 		Motion* p_v_target;
	// // 		p_v_target = GetMotion(0, true);
	// // 		this->mTargetPositions = p_v_target->GetPosition();
	// // 		this->mTargetVelocities = p_v_target->GetVelocity();

	// // 		std::cout<<p_v_target->GetPosition().transpose()<<std::endl<<std::endl;
	// // 		std::cout<<p_v_target->GetVelocity().transpose()<<std::endl<<std::endl;
	// // 		delete p_v_target;

	// // 		this->mPDTargetPositions = mTargetPositions;
	// // 		this->mPDTargetVelocities = mTargetVelocities;

	// // 		skel->setPositions(mTargetPositions);
	// // 		skel->setVelocities(mTargetVelocities);
	// // 		skel->computeForwardKinematics(true,true,false);

	// // 		this->mIsNanAtTerminal = false;
	// // 		this->mIsTerminal = false;
			
	// // 		// ClearRecord();
	// // 		// SaveStepInfo();

	// // 		// mRootZero = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	// // 		// mPrevPositions = mCharacter->GetSkeleton()->getPositions();
	// // 		mPrevTargetPositions = mTargetPositions;
			
	// // 		mPrevFrame = mCurrentFrame;
	// // 		mPrevFrame2 = mPrevFrame;
			
	// // 		mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	// // 		mTimeQueue.push(mCurrentFrame);
	// // 		mAdaptiveStep = 1;

	// 	}

// 		if(control_mode ==0 && mTime1+mBlendMargin >=blendFrame1){
// 			control_mode = 1;
// 			mRef2 = mSubControllers["FW_JUMP"]->mReferenceManager;
// 			mTime2 = blendFrame2- mBlendMargin;
// 			mBlendStep = 1;

// 			Eigen::Isometry3d prev_cycle_end= mRef1->GetRootTransform(mTime1+mBlendMargin, true);
// 			Eigen::Isometry3d cycle_start= mRef2->GetRootTransform(mTime2+mBlendMargin, true);

// 			mAlign2 = prev_cycle_end*cycle_start.inverse();
// 			Eigen::Vector3d p01 = dart::math::logMap(mAlign2.linear());			
// 			mAlign2.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));

// 			Eigen::Isometry3d cycle_start_edit = cycle_start;
// 			cycle_start_edit.linear() = mAlign2.linear().inverse()*prev_cycle_end.linear();
// 			mAlign2 = prev_cycle_end*cycle_start_edit.inverse();

// 			std::cout<<"blend @ "<<(mTime1+mBlendMargin)<<" , "<<(mTime2+mBlendMargin)<<std::endl;
// 		}
// 		else if(control_mode == 1 && (mTime1>= blendFrame1)){

// 			mCurrentController= mSubControllers["FW_JUMP"];
// 			control_mode = 2;
// 			mCurrentFrameOnPhase = mTime2- GetCurrentRefManager()->GetPhaseLength();
// 		}else if(control_mode == 2 && (mCurrentFrame-mBlendMargin>=blendFrame1)){
// 			mRef1 = mRef2;
// 			mAlign1 = mAlign2;
// 			mTime1 = mTime2;
// 			control_mode = 3;
				
// 			mRef2 = nullptr;
// 		}
		if(mCurrentFrame>=120){
			scenario_done =true;
			break;
		}
	}

	// this->mTiming.push_back(this->mController->GetCurrentFrame());
}

void MetaController::loadSceneObjects(std::string obj_path)
{
	std::cout<<"loadSceneObjects: "<<obj_path<<std::endl;
	mSceneObjects = std::vector<dart::dynamics::SkeletonPtr>();
	SkeletonBuilder::loadScene(obj_path, mSceneObjects);
	for(auto obj: mSceneObjects) this->mWorld->addSkeleton(obj);
	this->mLoadScene = true;
}

void MetaController::loadControllers(std::string ctrl_path)
{
	std::cout<<"loadControllers: "<<ctrl_path<<std::endl;
	TiXmlDocument doc;
	if(!doc.LoadFile(ctrl_path)){
		std::cout << "Can't open scene file : " << ctrl_path << std::endl;
	}

	TiXmlElement *skeldoc = doc.FirstChildElement("ControllerList");
	
	for(TiXmlElement *body = skeldoc->FirstChildElement("SubController"); body != nullptr; body = body->NextSiblingElement("SubController")){
		
		std::string ctrl_type = body->Attribute("type");
		std::string ctrl_bvh = body->Attribute("bvh");
		std::string ctrl_reg = body->Attribute("reg");
		std::string ctrl_ppo = body->Attribute("ppo");

		std::cout<< "================ ADD SUB Controller: "<<ctrl_type<<" :: "<<ctrl_bvh<<" , "<<ctrl_reg<<" , "<<ctrl_ppo<<std::endl;
		SubController* newSC;
		if(ctrl_type== "FW_JUMP"){
			newSC = new FW_JUMP_Controller(this, ctrl_bvh, ctrl_reg, ctrl_ppo);
		}else if(ctrl_type == "WALL_JUMP"){
			newSC = new WALL_JUMP_Controller(this, ctrl_bvh, ctrl_reg, ctrl_ppo);
		}else{
			std::cout<<" NOT A PROPER COTNROLLER TYPE : "<<ctrl_type<<std::endl;
			continue;
		}
		addSubController(newSC);	
	}
		// std::string ctrl_bvh = std::string(CAR_DIR)+std::string("/character/") + std::string(object_type) + std::string(".xml");
		// Eigen::VectorXd pos = string_to_vectorXd(body->Attribute("pos"));
}

// 공통
void MetaController::SetAction(const Eigen::VectorXd& action)
{
	this->mActions = action;
}

Eigen::VectorXd MetaController::GetState()
{
	// 1) 공통 ... 
	// 2) according to mCurrentController...

	bool isAdaptive= true; bool mRecord = true;
	if(mIsTerminal && terminationReason != 8){
		return Eigen::VectorXd::Zero(mNumState);
	}
	auto& skel = mCharacter->GetSkeleton();
	
	double root_height = skel->getRootBodyNode()->getCOM()[1];

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	Eigen::VectorXd p,v;
	// p.resize(p_save.rows()-6);
	// p = p_save.tail(p_save.rows()-6);

	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	int num_p = (n_bnodes - 1) * 6;
	p.resize(num_p);

	for(int i = 1; i < n_bnodes; i++){
		Eigen::Isometry3d transform = skel->getBodyNode(i)->getRelativeTransform();
		// Eigen::Quaterniond q(transform.linear());
		p.segment<6>(6*(i-1)) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
								 transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2);
	}

	v= v_save;

	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();
	Eigen::VectorXd ee;
	ee.resize(mEndEffectors.size()*3);
	for(int i=0;i<mEndEffectors.size();i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee.segment<3>(3*i) << transform.translation();
	}
	double t = GetCurrentRefManager()->GetTimeStep(mCurrentFrameOnPhase, isAdaptive);

	Motion* p_v_target = GetMotion(t, true);

	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_v_target->GetPosition(), p_v_target->GetVelocity()*t);

	delete p_v_target;

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);
	Eigen::VectorXd state;


	/// 2) according to mCurrentController
	Eigen::VectorXd param = mCurrentController->GetParamGoal();
	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2+param.rows());
	state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase, param;

	// if(control_mode == 1 && mCurrentFrameOnPhase< 1){
	// 	std::cout<<"@ "<<mCurrentFrameOnPhase<<std::endl;
	// 	std::cout<<"p: "<<p.transpose()<<std::endl;
	// 	std::cout<<"v: "<<v.transpose()<<std::endl;
	// 	std::cout<<"up-vec: "<<up_vec_angle<<std::endl;
	// 	std::cout<<p_next.transpose()<<std::endl;
	// 	// std::cout<<"v.front : "<<v.head<6>().transpose()<<std::endl;
	// 	// std::cout<<"v.front : "<<v.segment<6>(6).transpose()<<std::endl;
	// 	std::cout<<"root_height : "<<root_height<<std::endl;
	// 	std::cout<<"mAdaptiveStep : "<<mAdaptiveStep<<std::endl;
	// 	// std::cout<<"mCurrentFrameOnPhase : "<<mCurrentFrameOnPhase<<std::endl;
	// 	std::cout<<"ee: "<<ee.transpose()<<std::endl;
	// 	std::cout<<"param : "<<param.transpose()<<" / "<<mCurrentController->mType<<std::endl;
	// 	std::cout<<std::endl;
	// }

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true, true, false);

	return state;
	// if(isParametric) {
	// 	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2+mParamGoal.rows());
	// 	state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase, mParamGoal;
	// }
	// else {
	// 	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2);
	// 	state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase;
	// }

	// return state;
}


int MetaController::GetNumState()
{
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	int num_p = (n_bnodes - 1) * 6;
	int num_v = mCharacter->GetSkeleton()->getVelocities().rows();

	int num_p_next= mEndEffectors.size()*12+15;
	int num_ee = mEndEffectors.size() *3;

	// state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2+param.rows());

	return ( num_p + num_v + 1+1+ num_p_next + num_ee + 2);
}



void MetaController::Step()
{
	// 1) 공통 ... 
	// 2) according to mCurrentController...
	
	if(IsTerminalState())
		return;

	bool isAdaptive = true;
	bool mRecord= true;

	// Eigen::VectorXd s = this->GetState();

	Eigen::VectorXd a = mActions;

	// set action target pos
	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	for(int i = 0; i < mInterestedDof; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.2, -0.7*M_PI, 0.7*M_PI);
	}

	mActions[mInterestedDof] = dart::math::clip(mActions[mInterestedDof]*1.2, -2.0, 1.0);
	mActions[mInterestedDof] = exp(mActions[mInterestedDof]);
	mAdaptiveStep = mActions[mInterestedDof];
	// if(!isAdaptive) mAdaptiveStep = 1;

	mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame += mAdaptiveStep;
	this->mCurrentFrameOnPhase += mAdaptiveStep;

	mTime1 += mAdaptiveStep;
	mTime2 += mAdaptiveStep;
	mBlendStep++;

	// nTotalSteps += 1;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	// TODO : ALIGN / BLEND (if needed)
	Motion* p_v_target = GetMotion(0, true);
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = mCharacter->GetSkeleton()->getPositionDifferences(mTargetPositions, mPrevTargetPositions) / 0.033 * (mCurrentFrame - mPrevFrame);
	delete p_v_target;

	p_v_target = GetMotion(0, false);

	this->mPDTargetPositions = p_v_target->GetPosition();
	this->mPDTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	int count_dof = 0;

	for(int i = 1; i <= num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getIndexInSkeleton(0);
		int dof = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getNumDofs();
		mPDTargetPositions.block(idx, 0, dof, 1) += mActions.block(count_dof, 0, dof, 1);
		count_dof += dof;
	}
	
	for(int i = 0; i < this->mSimPerCon; i += 2){

		for(int j = 0; j < 2; j++) {
			//mCharacter->GetSkeleton()->setSPDTarget(mPDTargetPositions, 600, 49);

			if(mCurrentController== mSubControllers["WALL_JUMP"]){
				Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
				for(int j = 0; j < num_body_nodes; j++) {
					int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
					int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
					std::string name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
					double torquelim = mCharacter->GetTorqueLimit(name) * 1.5;
					double torque_norm = torque.block(idx, 0, dof, 1).norm();
				
					torque.block(idx, 0, dof, 1) = std::max(-torquelim, std::min(torquelim, torque_norm)) * torque.block(idx, 0, dof, 1).normalized();
				}

				mCharacter->GetSkeleton()->setForces(torque);
				mWorld->step(false);	

			}else{
				Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
				mCharacter->GetSkeleton()->setForces(torque);
				mWorld->step(false);
			}

				// Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
				// mCharacter->GetSkeleton()->setForces(torque);
				// mWorld->step(false);

		}

		mTimeElapsed += 2 * mAdaptiveStep;
	}

	mCurrentController->Step();

	mTiming.push_back(mCurrentFrame);

	this->UpdateTerminalInfo();

	if(mRecord) {
		SaveStepInfo();
	}

	mPrevTargetPositions = mTargetPositions;
	mPrevFrame = mCurrentFrame;

	if(mPosQueue.size() >= 3)
		mPosQueue.pop();
	if(mTimeQueue.size() >= 3)
		mTimeQueue.pop();
	mPosQueue.push(mCharacter->GetSkeleton()->getPositions());
	mTimeQueue.push(mCurrentFrame);

}

void MetaController::UpdateTerminalInfo()
{
	// TODO
	if(mCurrentController->IsTerminalState()) mIsTerminal = true;

	Eigen::VectorXd p_ideal = mTargetPositions;
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(mTargetPositions);
	skel->computeForwardKinematics(true,false,false);

	Eigen::Isometry3d root_diff = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	
	Eigen::AngleAxisd root_diff_aa(root_diff.linear());
	double angle = RadianClamp(root_diff_aa.angle());
	Eigen::Vector3d root_pos_diff = root_diff.translation();


	// check nan
	if(dart::math::isNan(p)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 3;
	} else if(dart::math::isNan(v)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 4;
	}
	//characterConfigration
	// else if(root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
	// 	mIsTerminal = true;
	// 	terminationReason = 2;
	// } else if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
	// 	mIsTerminal = true;
	// 	terminationReason = 1;
	// } else if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
	// 	mIsTerminal = true;
	// 	terminationReason = 5;
	// } 
	else if(scenario_done){
		mIsTerminal = true;
		terminationReason =  8;
	}

	// else if(mCurrentFrame > GetCurrentRefManager()->GetPhaseLength()) { 
	// 	mIsTerminal = true;
	// 	terminationReason =  8;
	// }
	// if(mRecord) {
		if(mIsTerminal) std::cout << "terminationReason : "<<terminationReason << std::endl;
	// }

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true,true,false);

}


Eigen::VectorXd 
MetaController::
GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel) {
	Eigen::VectorXd ret;
	auto& skel = mCharacter->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(pos);
	skel->setVelocities(vel);
	skel->computeForwardKinematics(true, true, false);

	ret.resize((num_ee)*12+15);
//	ret.resize((num_ee)*9+12);

	for(int i=0;i<num_ee;i++)
	{		
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		//Eigen::Quaterniond q(transform.linear());
		// Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
		ret.segment<9>(9*i) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
							   transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2), 
							   transform.translation();
//		ret.segment<6>(6*i) << rot, transform.translation();
	}


	for(int i=0;i<num_ee;i++)
	{
	    int idx = skel->getBodyNode(mEndEffectors[i])->getParentJoint()->getIndexInSkeleton(0);
		ret.segment<3>(9*num_ee + 3*i) << vel.segment<3>(idx);
//	    ret.segment<3>(6*num_ee + 3*i) << vel.segment<3>(idx);

	}

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	//Eigen::Quaterniond q(transform.linear());

	Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
	Eigen::Vector3d root_angular_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getAngularVelocity();
	Eigen::Vector3d root_linear_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getCOMLinearVelocity();

	ret.tail<15>() << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
					  transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2),
					  transform.translation(), root_angular_vel_relative, root_linear_vel_relative;
//	ret.tail<12>() << rot, transform.translation(), root_angular_vel_relative, root_linear_vel_relative;

	// restore
	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true, true, false);

	return ret;
}

void MetaController::SaveStepInfo()
{
	mRecordBVHPosition.push_back(GetCurrentRefManager()->GetPosition(mCurrentFrameOnPhase, false));
	mRecordTargetPosition.push_back(mTargetPositions);
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	mRecordCOM.push_back(mCharacter->GetSkeleton()->getCOM());
	mRecordPhase.push_back(mCurrentFrame);

	// if(mRecord) {
	// 	mRecordObjPosition.push_back(mObject->GetSkeleton()->getPositions());
	// }
	// bool rightContact = CheckCollisionWithGround("RightFoot") || CheckCollisionWithGround("RightToe");
	// bool leftContact = CheckCollisionWithGround("LeftFoot") || CheckCollisionWithGround("LeftToe");

	// mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));
}

void switchController(std::string type, int frame=-1)
{
	// 
}

Motion* MetaController::GetMotion(double t, bool isAdaptive){
	// isAdaptive: 
	// true: followMotion, 
	// false: PDMotion, 
	Motion * m ;
	if(mRef2== nullptr){
		m = mRef1->GetMotion(mTime1+t, isAdaptive);
		m->MultiplyRootTransform(mAlign1);
	}else{
		Motion* m1 = mRef1->GetMotion(mTime1+t, isAdaptive);
		m1->MultiplyRootTransform(mAlign1);

		Motion* m2 = mRef2->GetMotion(mTime2+t, isAdaptive);
		m2->MultiplyRootTransform(mAlign2);

		double blendRatio = (double)mBlendStep/ (2*(mBlendMargin+1));
		Eigen::VectorXd new_p = BlendPosition(m1->GetPosition(), m2->GetPosition(), blendRatio);
		// m->SetPosition(BlendPosition(m1->GetPosition(), m2->GetPosition(), blendRatio));

		// // next
		// Motion* m1_next = mRef1->GetMotion(mTime1+t+1, isAdaptive);
		// m1_next->MultiplyRootTransform(mAlign1);

		// Motion* m2_next = mRef2->GetMotion(mTime2+t+1, isAdaptive);
		// m2_next->MultiplyRootTransform(mAlign2);

		// double blendRatio_next= (double)(mBlendStep+1)/ (2*(mBlendMargin+1));
		// if(blendRatio_next > 1) blendRatio_next = 1;
		// Eigen::VectorXd new_p_next = BlendPosition(m1_next->GetPosition(), m2_next->GetPosition(), blendRatio_next);

		// Eigen::VectorXd new_v = mCharacter->GetSkeleton()->getPositionDifferences(new_p_next, new_p) / 0.033;
		// if(mBlendStep < mBlendMargin) new_v= m1->GetVelocity();
		// else new_v= m2->GetVelocity();

		Eigen::VectorXd new_v(m1->GetVelocity().rows());
		new_v= BlendPosition(m1->GetVelocity(), m2->GetVelocity(), blendRatio);
		m = new Motion(new_p, new_v);

		delete m1; delete m2; 
		// delete m1_next; delete m2_next;
   }

   return m;
}



} //end of namespace DPhy
