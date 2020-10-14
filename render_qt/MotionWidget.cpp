#include "MotionWidget.h"
#include "SkeletonBuilder.h"
#include "Character.h"
#include "Functions.h"
#include <GL/glu.h>
#include <iostream>
#include <Eigen/Geometry>
#include <chrono>
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

	v_param.resize(2);
    v_param.setZero();

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

    mSkel_bvh = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    mSkel_reg = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    mSkel_sim = DPhy::SkeletonBuilder::BuildFromFile(path).first;
	
	if(ppo == "") {
		mRunSim = false;
		mDrawSim = false;
	} else {
		mRunSim = true;
	}
	if(reg == "") {
		mRunReg = false;
		mDrawReg = false;
	} else {
		mRunReg = true;
	}

    path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
    DPhy::Character* ref = new DPhy::Character(path);
    mReferenceManager = new DPhy::ReferenceManager(ref);
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);
    
    if(mRunSim) {
	    path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(ppo, '/')[0] + std::string("/");
	    mReferenceManager->InitOptimization(1, path);
	    mReferenceManager->LoadAdaptiveMotion("");
	    mDrawReg = true;

    } else if(mRunReg) {
    	mReferenceManager->InitOptimization(1, "");
    }

    std::vector<Eigen::VectorXd> pos;
    double phase = 0;
    for(int i = 0; i < 500; i++) {
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
	        this->mRegression.attr("initRun")(path, mReferenceManager->GetTargetBase().rows() + 1, mReferenceManager->GetDOF() + 1);

	        path = path + "boundary";
	        char buffer[256];

	        std::ifstream is;
	        is.open(path);

	        int param_dof = v_param.rows();
	        while(!is.eof()) {
	            Eigen::VectorXd tp(param_dof);
	            for(int j = 0; j < param_dof; j++) {
	                is >> buffer;
	                tp[j] = atof(buffer);
	            }
	            //comma
	            is >> buffer;

	            Eigen::VectorXd tp2(param_dof);
	            for(int j = 0; j < param_dof; j++) {
	                is >> buffer;
	                tp2[j] = atof(buffer);
	            }
	            if((tp - tp2).norm() < 1e-2) 
	                break;
	            Eigen::VectorXd tp_mean = (tp + tp2) * 0.5;

	            mParamRange.push_back(tp_mean);

	        }
	        is.close();
	        std::stable_sort(mParamRange.begin(), mParamRange.end(), cmp);
    	}
    	if(ppo != "") {
    		this->mController = new DPhy::Controller(mReferenceManager, true, mRunReg, true);

    		p::object ppo_main = p::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + ppo;
			this->mPPO.attr("initRun")(path,
									   this->mController->GetNumState(), 
									   this->mController->GetNumAction());
			if(reg == "")
				RunPPO();
			
    	}
    
    } catch (const p::error_already_set&) {
        PyErr_Print();
    }    
}
void 
MotionWidget::
setValueX(const int &x){
    v_param(0) = x;
}
void 
MotionWidget::
setValueY(const int &y){
    v_param(1) = y;
}
void 
MotionWidget::
UpdateParam(const bool& pressed) {
	if(mRunReg) {
	    Eigen::VectorXd tp(v_param.rows());
	    int startIdx = 0, endIdx = mParamRange.size() - 1;
	    for(int i = 0 ; i < tp.rows(); i++) {
	        double min = mParamRange[startIdx](i);
	        double max = mParamRange[endIdx](i);
	        tp(i) = (max - min) * 0.1 * v_param(i) + min;

	        for(int j = startIdx; j <= endIdx + 1; j++) {
	            if(j == endIdx || mParamRange[j][i] > tp(i)) {
	                endIdx = j-1;
	                for(int k = endIdx; k >= startIdx; k--) {
	                    if(mParamRange[endIdx][i] != mParamRange[k][i]) {
	                        startIdx = k+1;
	                        break;
	                    }
	                }
	                break;
	            }
	        }
	    }
	    std::cout << "parameter updated: " << tp.transpose() << std::endl;
		Eigen::VectorXd tp_full = mReferenceManager->GetTargetFull();		
		Eigen::VectorXd tp_idx = mReferenceManager->GetTargetFeatureIdx();		
		for(int i = 0; i < tp_idx.size(); i++) {
			tp_full(tp_idx(i)) = tp(i);
		}

	    int dof = mReferenceManager->GetDOF() + 1;

	    std::vector<Eigen::VectorXd> cps;
	    for(int i = 0; i < mReferenceManager->GetNumCPS() ; i++) {
	        cps.push_back(Eigen::VectorXd::Zero(dof));
	    }

	    for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
	        Eigen::VectorXd input(mReferenceManager->GetTargetBase().rows() + 1);
	        input << j, tp;
	        p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
	    
	        np::ndarray na = np::from_object(a);
	        cps[j] = DPhy::toEigenVector(na, dof);
	    }

	    mReferenceManager->LoadAdaptiveMotion(cps);
	    if(!mRunSim) {
		    std::vector<Eigen::VectorXd> pos;
		    double phase = 0;
		    for(int i = 0; i < 500; i++) {
		        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        p(3) += 0.75;
		        pos.push_back(p);
		        phase += mReferenceManager->GetTimeStep(phase, true);
		    }
		    mTotalFrame = 500;
		    UpdateMotion(pos, 2);
	    } else {
	    	mController->SetTargetParameters(tp_full, tp);
			RunPPO();
	    }
	}
}
void
MotionWidget::
RunPPO() {
	std::vector<Eigen::VectorXd> pos_reg;
	std::vector<Eigen::VectorXd> pos_sim;

	int count = 0;
	mController->Reset(false);
	while(!this->mController->IsTerminalState()) {
		Eigen::VectorXd state = this->mController->GetState();

		p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
		np::ndarray na = np::from_object(a);
		Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

		this->mController->SetAction(action);
		this->mController->Step();
		count += 1;
	}

	for(int i = 0; i <= count; i++) {

		Eigen::VectorXd position = this->mController->GetPositions(i);
		Eigen::VectorXd position_reg = this->mController->GetTargetPositions(i);

		position(3) += 0.75;
		position_reg(3) += 0.75;

		pos_reg.push_back(position);
		pos_sim.push_back(position_reg);
	}

	UpdateMotion(pos_sim, 1);
	UpdateMotion(pos_reg, 2);

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
	}
	if(mDrawReg && n < mMotion_reg.size()) {
    	mSkel_reg->setPositions(mMotion_reg[n]);
	}
}
void
MotionWidget::
DrawSkeletons()
{

	if(mDrawBvh)
		GUI::DrawSkeleton(this->mSkel_bvh, 0);
	if(mDrawSim)
		GUI::DrawSkeleton(this->mSkel_sim, 0);
	if(mDrawReg)
		GUI::DrawSkeleton(this->mSkel_reg, 0);

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
	// Eigen::Vector3d com_root = mSkel->getRootBodyNode()->getCOM();
	// Eigen::Vector3d com_front = mSkel->getRootBodyNode()->getTransform()*Eigen::Vector3d(0.0, 0.0, 2.0);

	// if(this->mTrackCamera){
	// 	Eigen::Vector3d com = mSkel->getRootBodyNode()->getCOM();
	// 	Eigen::Isometry3d transform = mSkel->getRootBodyNode()->getTransform();
	// 	com[1] = 0.8;

	// 	Eigen::Vector3d camera_pos;
	// 	camera_pos << -3, 1, 1.5;
	// 	camera_pos = camera_pos + com;
	// 	camera_pos[1] = 2;

	// 	mCamera->SetCenter(com);
	// }
	// initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	// glEnable(GL_LIGHTING);
	// mCamera->Apply();

	DrawGround();
	DrawSkeletons();

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
// void
// MotionWidget::
// toggleDraw(int type) {
// 	if(type == 0) {
// 		mDrawBvh != mDrawBvh;
// 	}
// 	if(type == 1) {
// 		mDrawSim != mDrawSim;
// 	}
// 	if(type == 2) {
// 		mDrawReg != mDrawReg;
// 	}
// }
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
