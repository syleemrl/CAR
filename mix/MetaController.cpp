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
		loadScenario(std::string(CAR_DIR)+ std::string("/scene/")+scenario+std::string(".xml"));
		mCurrentTake = 0;
		mCurrentController = mSubControllers[mTakeList[mCurrentTake].ctrl_type];
		std::cout<<"INIT Controller: "<<mCurrentController->mType<<std::endl;
	}else{
		mCurrentController= mSubControllers["FW_JUMP"];
	}
	mPrevController = nullptr;

	mRef1 = mCurrentController->mReferenceManager;
	mTime1 = 0;
	if(mCurrentController == mSubControllers["RUN_SWING"]) mTime1= 1;
	mAlign1 = Eigen::Isometry3d::Identity();

	mRef2 = nullptr;
	mTime2 = 0;
	mAlign2 = Eigen::Isometry3d::Identity();

	// mTime1 = 4;
	// std::cout<<" set object : "<<mTakeList[mCurrentTake].target_object<<std::endl;
	// dart::dynamics::SkeletonPtr obj = mSceneObjects[mTakeList[mCurrentTake].target_object];
	// mCurrentController->setCurObject(obj);

	runScenario();
}

void MetaController::reset()
{
	this->mWorld->reset();
	auto& skel = mCharacter->GetSkeleton();
	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();

	bool isAdaptive = mCurrentController->mIsParametric;
	this->mCurrentFrame = mTime1; 
	this->mCurrentFrameOnPhase = std::fmod(mTime1, mCurrentController->mReferenceManager->GetPhaseLength());

	this->mTimeElapsed = 0;

	Motion* p_v_target;
	p_v_target = GetMotion(0, true);

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

				Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
				Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

				std::cout<<"@@@ "<<mCurrentFrame<<std::endl;
				for(int p_i=0; p_i<p.size(); p_i++) std::cout<<p[p_i]<<", ";
				std::cout<<"\n\n";

				for(int v_i=0; v_i<v.size(); v_i++) std::cout<<v[v_i]<<", ";
				std::cout<<"\n\n";
	
	// if(mTakeList[mCurrentTake].target_object!="") {
	// 	std::cout<<" set object : "<<mTakeList[mCurrentTake].target_object<<std::endl;

	// 	// dart::dynamics::SkeletonPtr obj = mSceneObjects[mTakeList[mCurrentTake].target_object];
	// 	Eigen::VectorXd p(obj->getNumDofs());
	// 	p.setZero();
	// 	obj->setPositions(p);

	// 	mCurrentController->setCurObject(obj);
	// }

	mCurrentController->reset();

}

Eigen::Isometry3d MetaController::calculateAlign(Eigen::Isometry3d cur, std::string from, double frame1, std::string to, double frame2){

	Eigen::Isometry3d prev_cycle_end= mSubControllers[from]->mReferenceManager->GetRootTransform(frame1, true);
	prev_cycle_end = cur*prev_cycle_end;

	Eigen::Isometry3d cycle_start= mSubControllers[to]->mReferenceManager->GetRootTransform(frame2, true);

	Eigen::Isometry3d align= Eigen::Isometry3d::Identity();

	Eigen::Vector3d prev_z= prev_cycle_end.linear()*Eigen::Vector3d::UnitZ();
	Eigen::Vector3d cur_z= cycle_start.linear()*Eigen::Vector3d::UnitZ();
	if(prev_z.dot(cur_z) < 0){
		align.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));
		align.translation() = prev_cycle_end.translation()+cycle_start.translation();
		align.translation()[1] = 0;	

		std::cout<<"diff / prev end "<<prev_cycle_end.translation().transpose()<<" / cycle start: "<<cycle_start.translation().transpose()<<" / align: "<<align.translation().transpose()<<std::endl;
	}else{
		align.linear() = Eigen::Matrix3d::Identity();
		align.translation() = prev_cycle_end.translation()-cycle_start.translation();
		align.translation()[1] = 0;
		std::cout<<"same / prev end "<<prev_cycle_end.translation().transpose()<<" / cycle start: "<<cycle_start.translation().transpose()<<" / align: "<<align.translation().transpose()<<std::endl;
	}

	return align;
}
void MetaController::handleTargetObject(int scene_number)
{	
	if(mTakeList[scene_number].target_object=="") return;

	dart::dynamics::SkeletonPtr obj = SkeletonBuilder::loadSingleObj(m_obj_path, mSceneObjects, mTakeList[scene_number].target_object);
	mWorld->addSkeleton(obj);

	SubController* sc= mSubControllers[mTakeList[scene_number].ctrl_type];

	std::cout<<" set object : "<<mTakeList[scene_number].target_object<<std::endl;

	if(mTakeList[scene_number].ctrl_type.compare("RUN_SWING")==0 && sc->mIsParametric){
		// swing bar specific code
		Eigen::Isometry3d newTransform = obj->getBodyNode(0)->getWorldTransform();			
		Eigen::VectorXd param = mTakeList[scene_number].goalParam;
		newTransform.translation()[1] = param[0];
		std::cout<<"swing bar height : "<<newTransform.translation()[1]<<std::endl;
		dart::dynamics::BodyNode* bn= obj->getBodyNode("Bar");
		dart::dynamics::BodyNode* parent= bn->getParentBodyNode();

		auto parent_props = parent->getParentJoint()->getJointProperties();
		parent_props.mT_ChildBodyToJoint = newTransform.inverse();
		parent->getParentJoint()->setProperties(parent_props);

		auto props = bn->getParentJoint()->getJointProperties();
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*newTransform;
		bn->getParentJoint()->setProperties(props);
	}

	if(mTakeList[scene_number].ctrl_type.compare("FW_JUMP")==0 && sc->mIsParametric){
		Eigen::VectorXd param = mTakeList[scene_number].goalParam;			
		Eigen::VectorXd pos_obj = obj->getPositions();
		int n_obs = (int) floor((param(0) - 0.6) * 10 / 2);

		double base = 0.15;
		for(int i = 0; i < n_obs; i++) {
			pos_obj(6+i) = base;
			base = pos_obj(6+i);
		} for (int i = n_obs; i < pos_obj.rows() - 7; i++) {
			pos_obj(6+i) = 0;
		}
		obj->setPositions(pos_obj);
	}

	if(mTakeList[scene_number].ctrl_type.compare("WALL_JUMP")==0 && sc->mIsParametric){
		Eigen::VectorXd param = mTakeList[scene_number].goalParam;
		double h_grow = param[0]- sc->mReferenceManager->GetParamDMM()[0];
	
		auto bn = obj->getBodyNode("Jump_Box");

		auto shape_old = bn->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get();
		auto box = dynamic_cast<dart::dynamics::BoxShape*>(shape_old);
		Eigen::Vector3d origin = box->getSize();

		DPhy::SkeletonBuilder::DeformBodyNode(obj, bn, std::make_tuple("Jump_Box", Eigen::Vector3d(1, (h_grow+0.9)/origin[1], 1), 1));
	}
	sc->setCurObject(obj);

	if(scene_number == 0) return;

	std::string from = mTakeList[scene_number-1].ctrl_type;
	std::string to = mTakeList[scene_number].ctrl_type;
	auto frame_from_to = mTransitionRules[std::make_pair(from, to)];
	int blendFrame1 = frame_from_to.first;
	int blendFrame2 = frame_from_to.second;

	if(obj->getJoint(0)->getNumDofs() == 6){
		Eigen::VectorXd p = obj->getPositions();
		double blendFrame2_phase = std::fmod(blendFrame2, mSubControllers[to]->mReferenceManager->GetPhaseLength());
		Eigen::Isometry3d align_obj = calculateAlign(mAlign1, from, (mTime1+ mBlendMargin), to, blendFrame2_phase);

		MultiplyRootTransform(p, align_obj, false);
		// if(to.compare("WALL_JUMP")==0) p[5]+=0.1;
		obj->setPositions(p);
	}
	if(to.compare("RUN_SWING")==0){
		// swing bar specific code
		Eigen::Isometry3d newTransform = obj->getBodyNode(0)->getWorldTransform();
		double blendFrame2_phase = std::fmod(blendFrame2, mSubControllers[to]->mReferenceManager->GetPhaseLength());
		Eigen::Isometry3d align_obj = calculateAlign(mAlign1, from, (mTime1+ mBlendMargin), to, blendFrame2_phase);
		newTransform = align_obj*newTransform;
		
		dart::dynamics::BodyNode* bn= obj->getBodyNode("Bar");
		dart::dynamics::BodyNode* parent= bn->getParentBodyNode();

		auto parent_props = parent->getParentJoint()->getJointProperties();
		parent_props.mT_ChildBodyToJoint = newTransform.inverse();
		parent->getParentJoint()->setProperties(parent_props);

		auto props = bn->getParentJoint()->getJointProperties();
		props.mT_ParentBodyToJoint = parent->getTransform().inverse()*newTransform;
		bn->getParentJoint()->setProperties(props);
	}
	if(to.compare("FW_JUMP")==0 && mSubControllers[to]->mIsParametric){
		Eigen::VectorXd param = mTakeList[scene_number].goalParam;
		
		Eigen::VectorXd pos_obj = obj->getPositions();
		int n_obs = (int) floor((param(0) - 0.6) * 10 / 2);
		// std::cout << (param(0) - 0.6) * 10 / 2 << " "<< n_obs << std::endl;

		double base = 0.15;
		for(int i = 0; i < n_obs; i++) {
			pos_obj(6+i) = base;
			base = pos_obj(6+i);
		} for (int i = n_obs; i < pos_obj.rows() - 7; i++) {
			pos_obj(6+i) = 0;
		}
		obj->setPositions(pos_obj);
	}

}


void MetaController::runScenario(){
	//TODO
	std::cout<<"mCurrent Controller Type : "<<mCurrentController->mType<<std::endl;

	this->reset();
	handleTargetObject(0);

	while(! IsTerminalState()){

		std::cout<<"\n@ "<<mCurrentFrame<<"/ @"<<mTime1;
		if(mRef2!=nullptr) std::cout<<" / "<<mTime2<<" / "<<((double)mBlendStep/2/(mBlendMargin+1));
		std::cout<<std::endl;
		Eigen::VectorXd state = GetState();

		p::object a = this->mCurrentController->mPPO.attr("run")(DPhy::toNumPyArray(state));

		np::ndarray na = np::from_object(a);
		Eigen::VectorXd action = DPhy::toEigenVector(na,this->GetNumAction());
		
		
		this->SetAction(action);

		this->Step();	
		
		mTransitionRules.emplace(std::make_pair("FW_JUMP", "WALL_JUMP"), std::make_pair(40, 33)); //13
		mTransitionRules.emplace(std::make_pair("WALL_JUMP", "FW_JUMP"), std::make_pair(103, 5)); //13

		mTransitionRules.emplace(std::make_pair("WALL_JUMP", "RUN_SWING"), std::make_pair(92, 23));
		mTransitionRules.emplace(std::make_pair("RUN_SWING", "WALL_JUMP"), std::make_pair(99, 29));

		mTransitionRules.emplace(std::make_pair("FW_JUMP", "RUN_SWING"), std::make_pair(42, 13)); // TODO
		mTransitionRules.emplace(std::make_pair("RUN_SWING", "FW_JUMP"), std::make_pair(95, 0));

		if(mCurrentTake+1 < mTakeList.size()){

			std::string from = mTakeList[mCurrentTake].ctrl_type;
			std::string to = mTakeList[mCurrentTake+1].ctrl_type;
			auto frame_from_to = mTransitionRules[std::make_pair(from, to)];
			int blendFrame1 = frame_from_to.first;
			int blendFrame2 = frame_from_to.second;


			// WITHOUT BLENDING
			// if(to=="RUN_CONNECT"){

			// 	if(mTime1 >= blendFrame1){

			// 		std::cout<<"TRANSITION :: "<<from<<" -> "<<to<<std::endl;

			// 		Eigen::Isometry3d prevAlign = mAlign1;
			// 		mAlign1 = calculateAlign(prevAlign, from, blendFrame1, to, blendFrame2);
			// 		std::cout<<mAlign1.linear()<<"\n"<<mAlign1.translation().transpose()<<"\n";

			// 		mCurrentController = mSubControllers[to];
			// 		mRef1 = mSubControllers[to]->mReferenceManager;
			// 		mTime1 = blendFrame2;
			// 		mCurrentFrameOnPhase = std::fmod(mTime1, mRef1->GetPhaseLength());
			// 		mCurrentTake ++;
			// 		mCurrentController->reset(mCurrentFrame, mCurrentFrameOnPhase);
					
			// 		// TODO				
			// 		if(mTakeList[mCurrentTake].target_object!="") {
			// 			std::cout<<" set object : "<<mTakeList[mCurrentTake].target_object<<std::endl;

			// 			dart::dynamics::SkeletonPtr obj = mSceneObjects[mTakeList[mCurrentTake].target_object];
			// 			Eigen::VectorXd p = obj->getPositions();
			// 			double blendFrame2_phase = std::fmod(blendFrame2, mSubControllers[to]->mReferenceManager->GetPhaseLength());
			// 			Eigen::Isometry3d align_obj = calculateAlign(prevAlign, from, blendFrame1, to, blendFrame2_phase);

			// 			MultiplyRootTransform(p, align_obj, false);
			// 			if(mCurrentTake==1) p[5]+=0.1;
			// 			obj->setPositions(p);

			// 			mCurrentController->setCurObject(obj);
			// 		}
			// 	}
			// }
			// else{

				double mTime1OnPhase = std::fmod(mTime1, mSubControllers[from]->mReferenceManager->GetPhaseLength());
				double mTime2OnPhase = std::fmod(mTime2, mSubControllers[to]->mReferenceManager->GetPhaseLength());
					
				if(control_mode ==0 && mTime1OnPhase+mBlendMargin >=blendFrame1){
					control_mode = 1;
					std::cout<<"TRANSITION :: "<<from<<" -> "<<to<<" :: "<<control_mode<<std::endl;

					if(mSubControllers[to]->mIsParametric) {
						mSubControllers[to]->mParamGoal = mTakeList[mCurrentTake+1].goalParam;
						mSubControllers[to]->mReferenceManager->SetParamGoal(mTakeList[mCurrentTake+1].goalParam);

				     	std::vector<Eigen::VectorXd> cps = mSubControllers[to]->mReferenceManager->GetRegressionMemory()->GetCPSFromNearestParams(mTakeList[mCurrentTake+1].goalParam);
					    mSubControllers[to]->mReferenceManager->LoadAdaptiveMotion(cps);
					}

					mRef2 = mSubControllers[to]->mReferenceManager;
					mTime2 = blendFrame2- mBlendMargin;
					if(mTime2 < 0) {
						mTime2 += mSubControllers[to]->mReferenceManager->GetPhaseLength();
						blendFrame2 += mSubControllers[to]->mReferenceManager->GetPhaseLength();
					}
					mTime2OnPhase = std::fmod(mTime2, mSubControllers[to]->mReferenceManager->GetPhaseLength());
					mSubControllers[to]->reset(mTime2, mTime2OnPhase);
					mBlendStep = 1;
					std::cout<<"blendFrame1 ; "<<blendFrame1<<" / mTime1+mBlendMargin: "<<(mTime1+mBlendMargin)<<std::endl;
					mAlign2 = calculateAlign(mAlign1, from, (mTime1+ mBlendMargin), to, blendFrame2);
					handleTargetObject(mCurrentTake+1);
				}
				else if(control_mode == 1 && (mTime1OnPhase>= blendFrame1)){
					mCurrentController= mSubControllers[to];
					control_mode = 2;
					mCurrentFrameOnPhase = mTime2OnPhase;
					std::cout<<"TRANSITION :: "<<from<<" -> "<<to<<" :: "<<control_mode<<"/ phase: "<<mCurrentFrameOnPhase<<std::endl;
				}else if(control_mode == 2 && (mTime2OnPhase>=blendFrame2+mBlendMargin)){
					mRef1 = mRef2;
					mAlign1 = mAlign2;
					mTime1 = mTime2;

					mCurrentTake++;
					control_mode = 0;
					std::cout<<"TRANSITION :: "<<from<<" -> "<<to<<" :: "<<control_mode<<"/ Goal :"<<mRef1->GetParamGoal().transpose()<<std::endl;

					mRef2 = nullptr;
				}
			}


			// }
		// if(mCurrentTake ==1) break;
		// if(mCurrentFrame>=120){
		// 	scenario_done= true;
		// 	break;
		// }
		if(mCurrentTake+1 == mTakeList.size() && mCurrentFrameOnPhase + 5 >= mRef1->GetPhaseLength()){
			scenario_done =true;
			break;
		}

	}
}

void MetaController::loadSceneObjects(std::string obj_path)
{
	// std::cout<<"loadSceneObjects: "<<obj_path<<std::endl;
	m_obj_path= obj_path;
	mSceneObjects = std::map<std::string, dart::dynamics::SkeletonPtr>();

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
		bool isParametric = true;

		if(body->Attribute("isParametric")!=nullptr){
			std::string par = body->Attribute("isParametric");
			if(par.find("false")!=std::string::npos) isParametric = false;
		}

		std::cout<< "================ ADD SUB Controller: "<<ctrl_type<<" :: "<<ctrl_bvh<<" , "<<ctrl_reg<<" , "<<ctrl_ppo<<" , isParametric: "<<isParametric<<std::endl;
		SubController* newSC;
		if(ctrl_type== "FW_JUMP"){
			newSC = new FW_JUMP_Controller(this, ctrl_bvh, ctrl_ppo, ctrl_reg, isParametric);
		}else if(ctrl_type == "WALL_JUMP"){
			newSC = new WALL_JUMP_Controller(this, ctrl_bvh, ctrl_ppo, ctrl_reg, isParametric);
		}else if(ctrl_type == "RUN_SWING"){
			newSC = new RUN_SWING_Controller(this, ctrl_bvh, ctrl_ppo, ctrl_reg, isParametric);
		}else if(ctrl_type == "RUN_CONNECT"){
			newSC = new RUN_CONNECT_Controller(this, ctrl_bvh, ctrl_ppo, ctrl_reg, isParametric);	
		}else{
			std::cout<<" NOT A PROPER COTNROLLER TYPE : "<<ctrl_type<<std::endl;
			continue;
		}

		if(isParametric) {
			newSC->mParamGoal = newSC->mReferenceManager->GetParamGoal();
		}
		addSubController(newSC);
	}
		// std::string ctrl_bvh = std::string(CAR_DIR)+std::string("/character/") + std::string(object_type) + std::string(".xml");
		// Eigen::VectorXd pos = string_to_vectorXd(body->Attribute("pos"));
}

void MetaController::loadScenario(std::string file_path)
{
	std::cout<<"loadScenario: "<<file_path<<std::endl;
	TiXmlDocument doc;
	if(!doc.LoadFile(file_path)){
		std::cout << "Can't open scenario file : " << file_path << std::endl;
	}

	TiXmlElement *skeldoc = doc.FirstChildElement("Scenario");
	
	for(TiXmlElement *body = skeldoc->FirstChildElement("Take"); body != nullptr; body = body->NextSiblingElement("Take")){
		
		std::string ctrl_type = body->Attribute("type");
		Eigen::VectorXd goal_param = (body->Attribute("goal_param")!=nullptr)? (DPhy::string_to_vectorXd(body->Attribute("goal_param"))) : Eigen::VectorXd(0);
		std::string target_object = (body->Attribute("target_object")!=nullptr)? (body->Attribute("target_object")) : "";

		Take new_take(ctrl_type);
		new_take.goalParam= goal_param;
		new_take.target_object = target_object;
		mTakeList.push_back(new_take);

		std::cout<<"===================== NEW TAKE : "<<ctrl_type<<" / "<<goal_param.transpose()<<" / target_object: "<<target_object<<" ====================="<<std::endl;
	}
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

	bool isAdaptive= mCurrentController->mIsParametric; bool mRecord = true;
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

	// if(mCurrentController->mType == "WALL_JUMP"){
		// std::cout<<"@@ "<<mTime1<<std::endl;
		// std::cout<<"p : "<<p.transpose()<<std::endl;
		// std::cout<<"v : "<<v.transpose()<<std::endl;
		// std::cout<<"up_vec_angle : "<<up_vec_angle<<std::endl;
		// std::cout<<"root_height : "<<root_height<<std::endl;
		// std::cout<<"p_next : "<<p_next.transpose()<<std::endl;
		// std::cout<<"mAdaptiveStep : "<<mAdaptiveStep<<std::endl;
		// std::cout<<"ee : "<<ee.transpose()<<std::endl;
		// std::cout<<"mCurrentFrameOnPhase : "<<mCurrentFrameOnPhase<<std::endl;
		// std::cout<<std::endl;
	// }

	/// 2) according to mCurrentController
	if(mCurrentController->mIsParametric){
		Eigen::VectorXd param = mCurrentController->GetParamGoal();
		state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2+param.rows());
		std::cout<<"@ "<<mCurrentFrame<<" / "<<mCurrentFrameOnPhase<<" / goal ; "<<param.transpose()<<std::endl;
		state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase, param;		
	}else{
		state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2);
		state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase;		
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	skel->computeForwardKinematics(true, true, false);

	return state;
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

	bool isAdaptive = mCurrentController->mIsParametric;
	bool mRecord= true;

	// Eigen::VectorXd s = this->GetState();
	std::cout<<"@ "<<mCurrentFrame<<std::endl;
	for(auto obj: mSceneObjects) {
		std::cout<<obj.first<<" / "<<obj.second->getBodyNode(0)->getWorldTransform().translation().transpose()<<std::endl;
	}

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
	if(! mCurrentController->mIsParametric) mAdaptiveStep = 1;

	mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame += mAdaptiveStep;
	this->mCurrentFrameOnPhase += mAdaptiveStep;

	mTime1 += mAdaptiveStep;
	mTime2 += mAdaptiveStep;
	mBlendStep+= mAdaptiveStep;

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

			if(mCurrentController!= mSubControllers["FW_JUMP"]){
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
	if(mCurrentFrameOnPhase >= mCurrentController->mReferenceManager->GetPhaseLength()){
		mCurrentFrameOnPhase -= mCurrentController->mReferenceManager->GetPhaseLength();
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
	else if(root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 2;
		std::cout<<p.segment<3>(3).transpose()<<" / target: "<<mTargetPositions.segment<3>(3).transpose()<<"/ dist: "<<root_pos_diff.norm()<<std::endl;
	} else if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
		mIsTerminal = true;
		terminationReason = 1;
	} else if(std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 5;
	} 
	else if(scenario_done){
		mIsTerminal = true;
		terminationReason =  8;
	}

	// else if(mCurrentFrame > 10) { 
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

		std::cout<<m1->GetPosition().segment<3>(3).transpose()<<" / "<<m2->GetPosition().segment<3>(3).transpose()<<" / "<<blendRatio<<std::endl;
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
