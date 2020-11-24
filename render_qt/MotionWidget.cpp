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
MotionWidget::
MotionWidget()
  :mCamera(new Camera(1000, 650)),mCurFrame(0),mPlay(false),
  mTrackCamera(false), mDrawBvh(true), mDrawSim(true), mDrawReg(true)
{
	this->startTimer(30);
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

	mSkel_obj = nullptr;
	#ifdef OBJECT_TYPE 
		std::cout<<"\nMotionWidget OBJ"<<std::endl;
		std::string object_path = std::string(CAR_DIR)+std::string("/character/") + std::string(OBJECT_TYPE) + std::string(".xml");
		mSkel_obj = DPhy::SkeletonBuilder::BuildFromFile(object_path).first;
	#endif
	
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

    if(mRunReg) {
    	mRegressionMemory = new DPhy::RegressionMemory();
		mReferenceManager->SetRegressionMemory(mRegressionMemory);

    }

    if(mRunSim) {
	    path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(ppo, '/')[0] + std::string("/");
	    std::cout<<"path : "<<path<<std::endl;
	    if(mRunReg)
	    	mReferenceManager->InitOptimization(1, path, true);
	    else
	    	mReferenceManager->InitOptimization(1, path);
	    mReferenceManager->LoadAdaptiveMotion("ref_1");
	    mDrawReg = true;

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
	        p(3) += (0.75 + 1.5); 
	        pos.push_back(p);
	        phase += mReferenceManager->GetTimeStep(phase, false);
    	}

   	 	UpdateMotion(pos, 3);
   	 	pos.clear();
    }

    phase = 0;
    for(int i = 0; i < 1000; i++) {
        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, false);
        p(3) -= 0.75; 
        pos.push_back(p);
        phase += mReferenceManager->GetTimeStep(phase, false);
    }

    UpdateMotion(pos, 0);
	initNetworkSetting(ppo, reg);

	DPhy::SetSkeletonColor(mSkel_bvh, Eigen::Vector4d(235./255., 73./255., 73./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_reg, Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_sim, Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_exp, Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));

	std::vector<int> check_frame = { 0, 33, 38}; // {0, 41, 45, 81};
	for(int cf: check_frame){
		mSkel_bvh->setPositions(mMotion_bvh[cf]);
		mSkel_bvh->computeForwardKinematics(true, false, false);
		Eigen::Vector3d root = mSkel_bvh->getPositions().segment<3>(3);
		Eigen::Vector3d left_foot = mSkel_bvh->getBodyNode("LeftFoot")->getWorldTransform().translation();
		Eigen::Vector3d right_foot = mSkel_bvh->getBodyNode("RightFoot")->getWorldTransform().translation();
		root[0]+=0.75; left_foot[0]+=0.75; right_foot[0]+=0.75;
		std::cout<<cf<<": "<<root.transpose()<<" / lf : "<<left_foot.transpose()<<" / rf : "<<right_foot.transpose()<<"/mid:"<<((left_foot+right_foot)/2.).transpose()<<std::endl;
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
	        mParamRange = mReferenceManager->GetParamRange();
	       
	        path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
		//	mRegressionMemory->SaveContinuousParamSpace(path + "param_cspace");
    	}
    	if(ppo != "") {
    		if (reg!="") this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
    		else this->mController = new DPhy::Controller(mReferenceManager, true, false, true); //adaptive=true, bool parametric=true, bool record=true
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

void MotionWidget::updateIthParam(int i)
{
    mReferenceManager->LoadAdaptiveMotion(mRegressionMemory->mloadAllSamples[i]->cps);		


    std::vector<Eigen::VectorXd> pos;
    double phase = 0;

    bool flag = false;
    for(int i = 0; i < 500; i++) {

        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
        p(3) += 0.75;
      	pos.push_back(p);
        //phase += mReferenceManager->GetTimeStep(phase, true);
        phase += 1;
    }
    mTotalFrame = 500;
    Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
	root_bvh(3) += 0.75;
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
		this->updateIthParam(regMemShow_idx);
	}
}

void 
MotionWidget::
UpdateNextParam(const bool& pressed) {
	//mRegressionMemory->LoadParamSpace(path + "param_space");
	if(mRunReg) {
		regMemShow_idx = (regMemShow_idx+1) % mRegressionMemory->mloadAllSamples.size();
		std::cout<<regMemShow_idx<<" / "<<(mRegressionMemory->mloadAllSamples.size())<<std::endl;
		this->updateIthParam(regMemShow_idx);
	}
}

void 
MotionWidget::
UpdateParam(const bool& pressed) {
	if(mRunReg) {
		Eigen::VectorXd tp(2);
		std::cout<<"MW / "<<v_param.transpose()<<std::endl;
		tp << v_param(0)*0.1, v_param(1)*0.1;
	    tp = mRegressionMemory->GetNearestParams(tp, 1)[0].second->param_normalized;
	    Eigen::VectorXd tp_denorm = mRegressionMemory->Denormalize(tp);
	    int dof = mReferenceManager->GetDOF() + 1;
	    double d = mRegressionMemory->GetDensity(tp);
	    std::cout << tp.transpose() <<"/"<<tp_denorm.transpose()<< " / d: " << d << std::endl;


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
		    ///// REGRESSION MOTION UPDATE
		    std::vector<Eigen::VectorXd> pos;
		    double phase = 0;
		    bool flag = false;
		    for(int i = 0; i < 500; i++) {

		        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        p(3) += 0.75;
		      	pos.push_back(p);
		        //phase += mReferenceManager->GetTimeStep(phase, true);
		        // if(i==41){
		        // 	Eigen::VectorXd pos_restore = mSkel_reg->GetPositions();
		        // 	mSkel_reg->setPositions(p);
		        // 	Eigen::Vector3d
		        // }
		        phase += 1;
		    }
		    mTotalFrame = 500;
		    Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
			root_bvh(3) += 0.75;
			pos = DPhy::Align(pos, root_bvh);

		    UpdateMotion(pos, 2);


		    //// BLEND MOTION UPDATE
		    pos.clear();
		   	std::vector<Eigen::VectorXd> cps = mRegressionMemory->GetCPSFromNearestParams(tp_denorm);
		    mReferenceManager->LoadAdaptiveMotion(cps);

		    phase = 0;
		    flag = false;
		    for(int i = 0; i < 500; i++) {
		        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        p(3) += (0.75 + 1.5); 
		        pos.push_back(p);
		        // phase += mReferenceManager->GetTimeStep(phase, true);
		       	phase += 1;
 
	    	}
			root_bvh(3) += 1.5;
			pos = DPhy::Align(pos, root_bvh);

	   	 	UpdateMotion(pos, 3);
	    } else {
	     	mTotalFrame = 0;
	     	mController->SetGoalParameters(tp_denorm);
		    // std::vector<Eigen::VectorXd> cps = mRegressionMemory->GetCPSFromNearestParams(tp_denorm);
		    // mReferenceManager->LoadAdaptiveMotion(cps);
			RunPPO();
	    }


		mPoints= Eigen::Vector3d(0, tp_denorm[0], tp_denorm[1]); //height: min(foot), distance(com)
		mPoints[0]+=2.25;

	    Eigen::VectorXd restore = mSkel_exp->getPositions();
	    Eigen::VectorXd targetPose= mReferenceManager->GetPosition(41, true);
		mSkel_exp->setPositions(targetPose);
		mSkel_exp->computeForwardKinematics(true, false, false);
		Eigen::Vector3d lf= mSkel_exp->getBodyNode("LeftFoot")->getWorldTransform().translation();
		Eigen::Vector3d rf= mSkel_exp->getBodyNode("RightFoot")->getWorldTransform().translation();
		std::cout<<"jumped : "<<std::min(lf[1], rf[1])<<" "<<mSkel_exp->getPositions()[5]<<std::endl;

		mSkel_exp->setPositions(restore);
		mSkel_exp->computeForwardKinematics(true, false, false);
	}

}
void
MotionWidget::
RunPPO() {
	std::vector<Eigen::VectorXd> pos_bvh;
	std::vector<Eigen::VectorXd> pos_reg;
	std::vector<Eigen::VectorXd> pos_sim;
	std::vector<Eigen::VectorXd> pos_obj;

	int count = 0;
	mController->Reset(false);
	this->mTiming= std::vector<double>();

	while(!this->mController->IsTerminalState()) {
		Eigen::VectorXd state = this->mController->GetState();

		p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
		np::ndarray na = np::from_object(a);
		Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

		this->mController->SetAction(action);
		this->mController->Step();
		this->mTiming.push_back(this->mController->GetCurrentLength());

		count += 1;
	}

	for(int i = 0; i <= count; i++) {

		Eigen::VectorXd position = this->mController->GetPositions(i);
		Eigen::VectorXd position_reg = this->mController->GetTargetPositions(i);
		Eigen::VectorXd position_bvh = this->mController->GetBVHPositions(i);

		Eigen::VectorXd position_obj = this->mController->GetObjPositions(i);


		position(3) += 0.75;
		position_reg(3) += 0.75;
		position_bvh(3) -= 0.75;

		pos_reg.push_back(position_reg);
		pos_sim.push_back(position);
		pos_bvh.push_back(position_bvh);
		
		#ifdef OBJECT_TYPE
		pos_obj.push_back(position_obj);
		#endif
	}
	// Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
	// root_bvh(3) += 0.75;
	// pos_sim =  DPhy::Align(pos_sim, root_bvh);
	// pos_reg =  DPhy::Align(pos_reg, root_bvh);
	UpdateMotion(pos_bvh, 0);
	UpdateMotion(pos_sim, 1);
	UpdateMotion(pos_reg, 2);
	#ifdef OBJECT_TYPE
	UpdateMotion(pos_obj, 5);
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
    	#ifdef OBJECT_TYPE
	    	mSkel_obj->setPositions(mMotion_obj[n]);
    	#endif
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
		GUI::DrawSkeleton(this->mSkel_bvh, 0);
		if(this->mSkel_obj) {
			glPushMatrix();
			glTranslated(-0.75, 0, 0);
			GUI::DrawSkeleton(this->mSkel_obj, 0);
			glPopMatrix();
		}
	}

	if(mDrawSim) {
		GUI::DrawSkeleton(this->mSkel_sim, 0);
		if(this->mSkel_obj) {
			glPushMatrix();
			glTranslated(0.75, 0, 0);
			GUI::DrawSkeleton(this->mSkel_obj, 0);
			glPopMatrix();
		}
	}
	if(mDrawReg) {
		GUI::DrawSkeleton(this->mSkel_reg, 0);
		// if(!mRunSim) {
		GUI::DrawPoint(mPoints, Eigen::Vector3d(1.0, 0.0, 0.0), 10);
		if(this->mSkel_obj) {
			glPushMatrix();
			glTranslated(0.75, 0, 0);
			GUI::DrawSkeleton(this->mSkel_obj, 0);
			glPopMatrix();
		}
	}
	if(mDrawExp) {
		GUI::DrawSkeleton(this->mSkel_exp, 0);
		GUI::DrawPoint(mPoints_exp, Eigen::Vector3d(1.0, 0.0, 0.0), 10);
		if(this->mSkel_obj) {
			glPushMatrix();
			glTranslated(2.25, 0, 0);
			GUI::DrawSkeleton(this->mSkel_obj, 0);
			glPopMatrix();
		}
	}

}	
void
MotionWidget::
DrawGround()
{
	Eigen::Vector3d com_root;
	com_root = this->mSkel_bvh->getRootBodyNode()->getCOM();
	if(mRunReg) {
		com_root = 0.5 * com_root + 0.5 * this->mSkel_reg->getRootBodyNode()->getCOM();
	} else if(mRunSim) {
			com_root = 0.5 * com_root + 0.5 * this->mSkel_sim->getRootBodyNode()->getCOM();	
	}
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
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
	} else if(type == 4) {
		mMotion_points = motion;
	} else if(type == 5){
		mMotion_obj = motion;
	}
	mCurFrame = 0;
	if(mTotalFrame == 0)
		mTotalFrame = motion.size();
	else if(mTotalFrame > motion.size()) {
		mTotalFrame = motion.size();
	}
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
ResetController()
{
    try {
    	if(mRunSim) {
    		this->mController->Reset(false);
			mController->SetGoalParameters(mReferenceManager->GetParamCur());
			RunPPO();
    	}    
    } catch (const p::error_already_set&) {
        PyErr_Print();
    }    
}
