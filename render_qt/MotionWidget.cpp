#include "MotionWidget.h"
#include "SkeletonBuilder.h"
#include "Character.h"
#include "Functions.h"
#include <GL/glu.h>
#include <iostream>
#include <Eigen/Geometry>
#include <QSlider>
#include <chrono>
#include <algorithm>
#include <ctime>
MotionWidget::
MotionWidget()
  :mCamera(new Camera(1000, 650)),mCurFrame(0),mPlay(false),
  mTrackCamera(false), mDrawBvh(true), mDrawSim(true), mDrawReg(true), mRD(), mMT(mRD()), mUniform(0.0, 1.0)
{
	this->startTimer(30);
}


void MotionWidget::
setRunBoxPosition(int box_series_idx, int box_idx, Eigen::Vector3d new_position){

	auto& skel = (box_series_idx == 0)? mSkel_obj: mSkel_obj_next;
	auto bn = (box_idx == 1)? skel->getBodyNode("Box1") : skel->getBodyNode("Box2");
	
	Eigen::Isometry3d newTransform = Eigen::Isometry3d::Identity();
	newTransform.translation() = new_position; 
	auto props = bn->getParentJoint()->getJointProperties();
	props.mT_ChildBodyToJoint = newTransform.inverse();
	bn->getParentJoint()->setProperties(props);
}

MotionWidget::
MotionWidget(std::string motion, std::string ppo, std::string reg)
  :MotionWidget()
{
	mCurFrame = 0;
	mTotalFrame = 0;

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

    mSkel_bvh = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    mSkel_reg = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    mSkel_sim = DPhy::SkeletonBuilder::BuildFromFile(path).first;
	mSkel_exp = DPhy::SkeletonBuilder::BuildFromFile(path).first;

	// path = std::string(CAR_DIR)+std::string("/character/obstacle.xml");
	// mSkel_obj = DPhy::SkeletonBuilder::BuildFromFile(path).first;
	// Eigen::VectorXd pos_obj = mSkel_obj->getPositions();

	// mSkel_obj->setPositions(pos_obj);

	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(CAR_DIR)+std::string("/character/ground.xml")).first;

	if(ppo == "") {
		mRunSim = false;
		mDrawSim = false;
	} else {
		mRunSim = true;
	}
	if(reg == "") {
		mRunReg = false;
		mDrawReg = false;
		mDrawExp = false;
	} else {
		mRunReg = true;
		mDrawExp = true;
	}
	mPoints.setZero();
	mPoints_exp.setZero();

    path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
    DPhy::Character* ref = new DPhy::Character(path);
    mReferenceManager = new DPhy::ReferenceManager(ref);
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);


    // if(mRunReg) {
    	mRegressionMemory = new DPhy::RegressionMemory();
		mReferenceManager->SetRegressionMemory(mRegressionMemory);

    // }
    mPath = "";
    if(mRunSim) {
	    path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(ppo, '/')[0] + std::string("/");
	    if(mRunReg)
	    	mReferenceManager->InitOptimization(1, path, true);
	    else
	    	mReferenceManager->InitOptimization(1, path);
	    mReferenceManager->LoadAdaptiveMotion("ref_1");
	    mDrawReg = true;
	    mPath = path;
    } else if(mRunReg) {
    	mReferenceManager->InitOptimization(1, "", true);
    }

	v_param.resize(mReferenceManager->GetParamGoal().rows());
    v_param.setZero();

    std::vector<Eigen::VectorXd> pos;
    double phase = 0;
    if(mRunReg) {
	    for(int i = 0; i < 1000; i++) {
	        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, false);
	        pos.push_back(p);
	        phase += mReferenceManager->GetTimeStep(phase, false);
    	}

   	 	UpdateMotion(pos, 3);
   	 	pos.clear();
    }

    phase = 0;
    for(int i = 0; i < 1000; i++) {
        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, false);
        pos.push_back(p);
        phase += mReferenceManager->GetTimeStep(phase, false);
    }

    UpdateMotion(pos, 0);
	initNetworkSetting(ppo, reg);

	#ifdef OBJECT_TYPE
		path = std::string(CAR_DIR)+std::string("/character/")+OBJECT_TYPE+std::string(".xml");
		mSkel_obj  = DPhy::SkeletonBuilder::BuildFromFile(path).first;
		mSkel_obj_next  = DPhy::SkeletonBuilder::BuildFromFile(path).first;

		double m_shift_height= 0;
		Eigen::Vector3d default_box0_pos= Eigen::Vector3d(0.0026,  -0.05+m_shift_height, -0.0935549);
		Eigen::Vector3d default_box1_pos= Eigen::Vector3d(0.0204, -0.05+m_shift_height,   1.26076);
		Eigen::Vector3d default_box2_pos= Eigen::Vector3d(0.0026,  -0.05+m_shift_height,   2.54534);

		Eigen::Vector3d box_shift= -(default_box2_pos-default_box0_pos);
		setRunBoxPosition(0, 1, default_box1_pos+box_shift);
		setRunBoxPosition(0, 2, default_box2_pos+box_shift);

		Eigen::Vector3d next_1_pos = default_box1_pos;
		Eigen::Vector3d next_2_pos = default_box2_pos;

		setRunBoxPosition(1, 1, next_1_pos);
		setRunBoxPosition(1, 2, next_2_pos);
	#endif


	DPhy::SetSkeletonColor(mSkel_bvh, Eigen::Vector4d(235./255., 73./255., 73./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_reg, Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_sim, Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_exp, Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));

	
	std::vector<int> check_frame = { 0, 11, 22, 33}; // {0, 41, 45, 81};
	for(int cf: check_frame){
		mSkel_bvh->setPositions(mMotion_bvh[cf]);
		mSkel_bvh->computeForwardKinematics(true, false, false);
		Eigen::Vector3d root = mSkel_bvh->getPositions().segment<3>(3);
		Eigen::Vector3d left_foot = mSkel_bvh->getBodyNode("LeftFoot")->getWorldTransform().translation();
		Eigen::Vector3d right_foot = mSkel_bvh->getBodyNode("RightFoot")->getWorldTransform().translation();

		Eigen::Vector3d left_toe = mSkel_bvh->getBodyNode("LeftToe")->getWorldTransform().translation();
		Eigen::Vector3d right_toe = mSkel_bvh->getBodyNode("RightToe")->getWorldTransform().translation();

		Eigen::Vector3d left_hand = mSkel_bvh->getBodyNode("LeftHand")->getWorldTransform().translation();
		Eigen::Vector3d right_hand = mSkel_bvh->getBodyNode("RightHand")->getWorldTransform().translation();

		// std::cout<<cf<<" lh ; "<<left_hand.transpose()<<"/ rh; "<<right_hand.transpose()<<std::endl;
		std::cout<<cf<<": "<<root.transpose()<<" / lf : "<<left_foot.transpose()<<" / rf : "<<right_foot.transpose()<<"/ mid:"<<((left_foot+right_foot)/2.).transpose()<<"/ toe: "<<left_toe.transpose()<<"/"<<right_toe.transpose()<<std::endl;
	}


}
bool cmp(const Eigen::VectorXd &p1, const Eigen::VectorXd &p2){
    for(int i = 0; i < p1.rows(); i++) {
        if(p1(i) < p2(i))
            return true;
        else if(p1(i) > p2(i))
            return false;
    }
    return false;
}
void
MotionWidget::
initNetworkSetting(std::string ppo, std::string reg) {

    Py_Initialize();
    np::initialize();
    try {
    	if(reg != "") {
			p::object reg_main = p::import("regression");
	        this->mRegression = reg_main.attr("Regression")();
	        std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
	        this->mRegression.attr("initRun")(path, mReferenceManager->GetParamGoal().rows() + 1, mReferenceManager->GetDOF() + 1);
			mRegressionMemory->LoadParamSpace(path + "param_space");
			std::cout << mRegressionMemory->GetVisitedRatio() << std::endl;
	        mParamRange = mReferenceManager->GetParamRange();
	       
	        path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
		//	mRegressionMemory->SaveContinuousParamSpace(path + "param_cspace");
    	}
    	if(ppo != "") {
    		if (reg!="") this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
    		else this->mController = new DPhy::Controller(mReferenceManager, false, false, true); //adaptive=true, bool parametric=true, bool record=true

    		// this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
			mController->SetGoalParameters(mReferenceManager->GetParamCur());

    		p::object ppo_main = p::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + ppo;
			this->mPPO.attr("initRun")(path,
									   this->mController->GetNumState(), 
									   this->mController->GetNumAction());
			RunPPO();
			
    	}
    
    } catch (const p::error_already_set&) {
        PyErr_Print();
    }    
}
void 
MotionWidget::
setValue(const int &x){
	auto slider = qobject_cast<QSlider*>(sender());
    auto i = slider->property("i").toInt();
    v_param(i) =  x;
}
void 
MotionWidget::
UpdateRandomParam(const bool& pressed) {

}
void 
MotionWidget::
UpdateParam(const bool& pressed) {


	if(mRunReg) {
		Eigen::VectorXd tp(mRegressionMemory->GetDim());
		tp = v_param*0.1;
	    
	    Eigen::VectorXd tp_denorm = mRegressionMemory->Denormalize(tp);
	    int dof = mReferenceManager->GetDOF() + 1;
	    double d = mRegressionMemory->GetDensity(tp);
	    std::cout << tp.transpose() << " " << tp_denorm.transpose() << " " << d << std::endl;

	    std::vector<Eigen::VectorXd> cps;
	    for(int i = 0; i < mReferenceManager->GetNumCPS() ; i++) {
	        cps.push_back(Eigen::VectorXd::Zero(dof));
	    }
	    for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
	        Eigen::VectorXd input(mRegressionMemory->GetDim() + 1);
	        input << j, tp;
	        p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
	    
	        np::ndarray na = np::from_object(a);
	        cps[j] = DPhy::toEigenVector(na, dof);
	    }

	    mReferenceManager->LoadAdaptiveMotion(cps);

	    double phase = 0;

	    if(!mRunSim) {
	    	#ifdef OBJECT_TYPE
	    		// TODO
			#endif

		    std::vector<Eigen::VectorXd> pos;
		    double phase = 0;

		    bool flag = false;
		    for(int i = 0; i < 500; i++) {

		        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        pos.push_back(p);
		        //phase += mReferenceManager->GetTimeStep(phase, true);
		        phase += 1;
		    }
		    mTotalFrame = 500;
		    Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
			pos = DPhy::Align(pos, root_bvh);

		    UpdateMotion(pos, 2);

		    pos.clear();
		   	std::vector<Eigen::VectorXd> cps = mRegressionMemory->GetCPSFromNearestParams(tp_denorm);
		    mReferenceManager->LoadAdaptiveMotion(cps);
		   
		    phase = 0;
		    flag = false;
		    for(int i = 0; i < 500; i++) {
		        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        pos.push_back(p);
		        // phase += mReferenceManager->GetTimeStep(phase, true);
		       	phase += 1;
 
	    	}
			pos = DPhy::Align(pos, root_bvh);

	   	 	UpdateMotion(pos, 3);
	    } else {
	     	mTotalFrame = 0;
	     	mController->SetGoalParameters(tp_denorm);
		    std::vector<Eigen::VectorXd> cps = mRegressionMemory->GetCPSFromNearestParams(tp_denorm);
		    mReferenceManager->LoadAdaptiveMotion(cps);
			RunPPO();
	    }
	}
}
void MotionWidget::UpdateIthParam(int i)
{
    mReferenceManager->LoadAdaptiveMotion(mRegressionMemory->mloadAllSamples[i]->cps);


    std::vector<Eigen::VectorXd> pos;
    double phase = 0;

    bool flag = false;
    for(int i = 0; i < 500; i++) {

        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
        
      pos.push_back(p);
        //phase += mReferenceManager->GetTimeStep(phase, true);
        phase += 1;
    }
    mTotalFrame = 500;
    Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
	pos = DPhy::Align(pos, root_bvh);

    UpdateMotion(pos, 2);
}
void
MotionWidget::
UpdatePrevParam(const bool& pressed) {
//mRegressionMemory->LoadParamSpace(path + "param_space");
	if(mRunReg) {
		if (regMemShow_idx == 0) regMemShow_idx = mRegressionMemory->mloadAllSamples.size()-1;
		else regMemShow_idx-- ;

		std::cout<<regMemShow_idx<<" / "<<(mRegressionMemory->mloadAllSamples.size())<<std::endl;
		this->UpdateIthParam(regMemShow_idx);
	}
}

void
MotionWidget::
UpdateNextParam(const bool& pressed) {
//mRegressionMemory->LoadParamSpace(path + "param_space");
	if(mRunReg) {
		regMemShow_idx = (regMemShow_idx+1) % mRegressionMemory->mloadAllSamples.size();
		std::cout<<regMemShow_idx<<" / "<<(mRegressionMemory->mloadAllSamples.size())<<std::endl;
		this->UpdateIthParam(regMemShow_idx);
	}
}


void
MotionWidget::
RunPPO() {
	std::vector<Eigen::VectorXd> pos_bvh;
	std::vector<Eigen::VectorXd> pos_reg;
	std::vector<Eigen::VectorXd> pos_sim;
	std::vector<Eigen::VectorXd> pos_obj;
	std::vector<Eigen::VectorXd> pos_obj_next;

	int count = 0;
	mController->Reset(false);
	this->mTiming= std::vector<double>();
	this->mTiming.push_back(this->mController->GetCurrentFrame());

	while(!this->mController->IsTerminalState()) {
		Eigen::VectorXd state = this->mController->GetState();

		p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
		np::ndarray na = np::from_object(a);
		Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

		this->mController->SetAction(action);
		this->mController->Step();
		this->mTiming.push_back(this->mController->GetCurrentFrame());

		count += 1;
	}

	for(int i = 0; i <= count; i++) {

		Eigen::VectorXd position = this->mController->GetPositions(i);
		Eigen::VectorXd position_reg = this->mController->GetTargetPositions(i);
		Eigen::VectorXd position_bvh = this->mController->GetBVHPositions(i);

		pos_reg.push_back(position_reg);
		pos_sim.push_back(position);
		pos_bvh.push_back(position_bvh);

		#ifdef OBJECT_TYPE
		Eigen::VectorXd position_obj = this->mController->GetObjPositions(i);
		// position_obj(3) += 0.75;
		pos_obj.push_back(position_obj);

		// Eigen::VectorXd position_obj_next = this->mController->GetObjPositions(1, i);
		// position_obj_next(3) += 0.75;
		// pos_obj_next.push_back(position_obj_next);

		#endif
	}
	Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
	pos_sim =  DPhy::Align(pos_sim, root_bvh);
	pos_reg =  DPhy::Align(pos_reg, root_bvh);
	UpdateMotion(pos_bvh, 0);
	UpdateMotion(pos_sim, 1);
	UpdateMotion(pos_reg, 2);

	std::cout<<"coutn? "<<count<<std::endl;
	#ifdef OBJECT_TYPE
	UpdateMotion(pos_obj, 4);
	// UpdateMotion(pos_obj_next, 5);
	#endif
}
void
MotionWidget::
initializeGL()
{
	glClearColor(1,1,1,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	mCamera->Apply();
}
void
MotionWidget::
resizeGL(int w,int h)
{
	glViewport(0, 0, w, h);
	mCamera->SetSize(w, h);
	mCamera->Apply();
}
void
MotionWidget::
SetFrame(int n)
{
	if(mDrawBvh && n < mMotion_bvh.size()) {
    	mSkel_bvh->setPositions(mMotion_bvh[n]);
	}
	if(mDrawSim && n < mMotion_sim.size()) {
    	mSkel_sim->setPositions(mMotion_sim[n]);

 		setRunBoxPosition(0,1, mMotion_obj[n].segment<3>(0));
		setRunBoxPosition(0,2, mMotion_obj[n].segment<3>(3));
		setRunBoxPosition(1,1, mMotion_obj[n].segment<3>(6));
		setRunBoxPosition(1,2, mMotion_obj[n].segment<3>(9));

    	// mSkel_obj->setPositions(mMotion_obj[n]);
    	// mSkel_obj_next->setPositions(mMotion_obj_next[n]);
	}
	if(mDrawReg && n < mMotion_reg.size()) {
    	mSkel_reg->setPositions(mMotion_reg[n]);
    	// mPoints = mMotion_points[n];
	}
	if(mDrawExp && n < mMotion_exp.size()) {
		mSkel_exp->setPositions(mMotion_exp[n]);
	}
}
void
MotionWidget::
DrawSkeletons()
{

	if(mDrawBvh){	
		glPushMatrix();	
		glTranslatef(-0.75, 0, 0);
		GUI::DrawSkeleton(this->mSkel_bvh, 0);
		glPopMatrix();	
	}
	if(mDrawSim) {
		glPushMatrix();	
		glTranslatef(0.75, 0, 0);
		GUI::DrawSkeleton(this->mSkel_sim, 0);
		GUI::DrawSkeleton(this->mSkel_obj, 0);
		GUI::DrawSkeleton(this->mSkel_obj_next, 0);
		glPopMatrix();	
		// GUI::DrawSkeleton(this->mSkel_obj, 0);
	}
	if(mDrawReg) {
		glPushMatrix();	
		glTranslatef(0.75, 0, 0);
		GUI::DrawSkeleton(this->mSkel_reg, 0);
		glPopMatrix();	
	}
	if(mDrawExp) {
		GUI::DrawSkeleton(this->mSkel_exp, 0);
		// if(!mRunSim)
		// 	GUI::DrawSkeleton(this->mSkel_obj, 0);

	}

}	
void
MotionWidget::
DrawGround()
{
	// GUI::DrawSkeleton(mGround, 0);
	double height = mGround->getBodyNode(0)->getWorldTransform().translation()[1];
	glPushMatrix();
	glTranslatef(0, height, 0);

		Eigen::Vector3d com_root;
		com_root = this->mSkel_bvh->getRootBodyNode()->getCOM();
		if(mRunReg) {
			com_root = 0.5 * com_root + 0.5 * this->mSkel_reg->getRootBodyNode()->getCOM();
		} else if(mRunSim) {
				com_root = 0.5 * com_root + 0.5 * this->mSkel_sim->getRootBodyNode()->getCOM();	
		}
		GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
	glPopMatrix();
}
void
MotionWidget::
paintGL()
{

	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	initLights();
	glEnable(GL_LIGHTING);

	mCamera->Apply();

	DrawGround();
	DrawSkeletons();

	if(mRunSim) GUI::DrawStringOnScreen(0.8, 0.9, std::to_string(mTiming[mCurFrame])+" / "+std::to_string(mCurFrame), true, Eigen::Vector3d::Zero());
	else GUI::DrawStringOnScreen(0.8, 0.9, std::to_string(mCurFrame), true, Eigen::Vector3d::Zero());
}
void
MotionWidget::
initLights()
{

	static float ambient[]           	 = {0.4, 0.4, 0.4, 1.0};
	static float diffuse[]             = {0.4, 0.4, 0.4, 1.0};
	static float front_mat_shininess[] = {60.0};
	static float front_mat_specular[]  = {0.2, 0.2,  0.2,  1.0};
	static float front_mat_diffuse[]   = {0.2, 0.2, 0.2, 1.0};
	static float lmodel_ambient[]      = {0.2, 0.2,  0.2,  1.0};
	static float lmodel_twoside[]      = {GL_TRUE};

	GLfloat position[] = {0.0, 1.0, 1.0, 0.0};
	GLfloat position1[] = {0.0, 1.0, -1.0, 0.0};

	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  lmodel_ambient);
	glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);

	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  front_mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   front_mat_diffuse);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);

	glEnable(GL_FOG);
	GLfloat fogColor[] = {200.0/256.0,200.0/256.0,200.0/256.0,1};
	glFogfv(GL_FOG_COLOR,fogColor);
	glFogi(GL_FOG_MODE,GL_LINEAR);
	glFogf(GL_FOG_DENSITY,0.05);
	glFogf(GL_FOG_START,20.0);
	glFogf(GL_FOG_END,40.0);
}
void
MotionWidget::
timerEvent(QTimerEvent* event)
{
	if(mPlay && mCurFrame < mTotalFrame) {
		mCurFrame += 1;
	} 
	SetFrame(this->mCurFrame);
	update();

}
void
MotionWidget::
toggleDrawBvh() {
	mDrawBvh = !mDrawBvh;

}
void
MotionWidget::
toggleDrawReg() {
	mDrawReg = !mDrawReg;

}
void
MotionWidget::
toggleDrawSim() {
	if(mRunSim)
		mDrawSim = !mDrawSim;

}
void
MotionWidget::
toggleDrawExp() {
	if(mRunReg)
		mDrawExp = !mDrawExp;

}
void
MotionWidget::
keyPressEvent(QKeyEvent *event)
{
	if(event->key() == Qt::Key_Escape){
		exit(0);
	}
	if(event->key() == Qt::Key_Space){
		mPlay = !mPlay;
		if(mPlay)
			std::cout << "Play." << std::endl;
		else 
			std::cout << "Pause." << std::endl;
	}
}
void
MotionWidget::
mousePressEvent(QMouseEvent* event)
{
	mIsDrag = true;
	mButton = event->button();
	mPrevX = event->x();
	mPrevY = event->y();
}
void
MotionWidget::
mouseMoveEvent(QMouseEvent* event)
{
	if(!mIsDrag)
	return;

	if (mButton == Qt::MidButton)
		mCamera->Translate(event->x(),event->y(),mPrevX,mPrevY);
	else if(mButton == Qt::LeftButton)
		mCamera->Rotate(event->x(),event->y(),mPrevX,mPrevY);

	mPrevX = event->x();
	mPrevY = event->y();
	update();
}
void
MotionWidget::
mouseReleaseEvent(QMouseEvent* event)
{
	mIsDrag = false;
	mButton = Qt::NoButton;
	update();
}
void
MotionWidget::
wheelEvent(QWheelEvent *event)
{
	if(event->angleDelta().y()>0)
	mCamera->Pan(0,-5,0,0);
	else
	mCamera->Pan(0,5,0,0);
	update();
}
void 
MotionWidget::
UpdateMotion(std::vector<Eigen::VectorXd> motion, int type)
{
	if(type == 0) {
		mMotion_bvh = motion;	
	}
	else if(type == 1) {
		mMotion_sim = motion;		
	}
	else if(type == 2) {
		mMotion_reg = motion;	
	}
	else if(type == 3) {
		mMotion_exp = motion;	
	} 
	else if(type == 4) {
		mMotion_obj = motion;
	}else if(type == 5) {
		mMotion_obj_next = motion;
	}
	mCurFrame = 0;
	if(mTotalFrame == 0)
		mTotalFrame = motion.size();
	else if(mTotalFrame > motion.size()) {
		mTotalFrame = motion.size();
	}
	std::cout<<"mTotalFrame: "<<mTotalFrame<<std::endl;
}
void
MotionWidget::
NextFrame()
{ 
	if(!mPlay) {
		this->mCurFrame += 1;
		this->SetFrame(this->mCurFrame);
	}
}
void
MotionWidget::
PrevFrame()
{
	if(!mPlay && mCurFrame > 0) {
		this->mCurFrame -= 1;
		this->SetFrame(this->mCurFrame);
	}
}
void
MotionWidget::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);
}
void 
MotionWidget::
togglePlay() {
	mPlay = !mPlay;
}
void 
MotionWidget::
Save() {
	time_t t;
	time(&t);

	std::string time_str = std::ctime(&t);

	mController->SaveDisplayedData(mPath + "record_" + time_str, true);
}
