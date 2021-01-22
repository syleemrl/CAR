#include "SceneMotionWidget.h"
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
SceneMotionWidget::
SceneMotionWidget()
  :mCamera(new Camera(1000, 650)),mCurFrame(0),mPlay(false),
  mTrackCamera(true),  mDrawSim(true), mDrawReg(true), mRD(), mMT(mRD()), mUniform(0.0, 1.0)
{
	this->startTimer(30);
	mCurFrame = 0;
	mTotalFrame = 0;
	mMC = new DPhy::MetaController();
	mMC->Reset();
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
    mSkel_sim = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    mSkel_reg = DPhy::SkeletonBuilder::BuildFromFile(path).first;

	DPhy::SetSkeletonColor(mSkel_sim, Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_reg, Eigen::Vector4d(235./255., 235./255., 0./255., 1.0));
	Record();

	this->installEventFilter(this);
	setFocusPolicy( Qt::StrongFocus );

}
void
SceneMotionWidget::
Record() {
	mMotion_reg.push_back(this->mMC->GetCurrentRefPositions());
    mMotion_sim.push_back(this->mMC->GetCurrentSimPositions());
    this->mTotalFrame++;
}
void
SceneMotionWidget::
Step() {
	for(int i = 0 ; i <= 2; i++) {
		this->mMC->Step();		
		Record();
	}
}
void
SceneMotionWidget::
initializeGL()
{
	glClearColor(1,1,1,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	mCamera->Apply();
}
void
SceneMotionWidget::
resizeGL(int w,int h)
{
	glViewport(0, 0, w, h);
	mCamera->SetSize(w, h);
	mCamera->Apply();
}
void
SceneMotionWidget::
SetFrame(int n)
{
	if(mDrawReg && n < mMotion_reg.size()) mSkel_reg->setPositions(mMotion_reg[n]);
	if(mDrawSim && n < mMotion_sim.size()) mSkel_sim->setPositions(mMotion_sim[n]);
}
void
SceneMotionWidget::
DrawSkeletons()
{
	if(mDrawSim)
		GUI::DrawSkeleton(this->mSkel_sim, 0);
	if(mDrawReg)
		GUI::DrawSkeleton(this->mSkel_reg, 0);
}	
void 
SceneMotionWidget::
RE()
{

	mCurFrame = 0;
	mTotalFrame = 0;
	mMotion_reg.clear();
	mMotion_sim.clear();

	mMC->Reset();
	Record();

}
void
SceneMotionWidget::
DrawGround()
{
	GUI::DrawGround(0, 0, 0);
	// Eigen::Vector3d com_root;
	// com_root = this->mSkel_bvh->getRootBodyNode()->getCOM();
	// if(mRunReg) {
	// 	com_root = 0.5 * com_root + 0.5 * this->mSkel_reg->getRootBodyNode()->getCOM();
	// } else if(mRunSim) {
	// 		com_root = 0.5 * com_root + 0.5 * this->mSkel_sim->getRootBodyNode()->getCOM();	
	// }
	// GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
SceneMotionWidget::
paintGL()
{

	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	initLights();
	glEnable(GL_LIGHTING);

	std::vector<Eigen::Vector3d> hpoints = mMC->GetHitPoints();
	for(int i = 0; i < hpoints.size(); i++) {
		GUI::DrawPoint(hpoints[i], Eigen::Vector3d(1.0, 0.0, 0.0), 10);
	}
	// if(mTrackCamera){
	// 	Eigen::Vector3d com = mMC->GetCOM();
	// 	// Eigen::Vector6d rootpos = mMC->GetCurrentSimPositions().segment<6>(0);
	// 	// Eigen::Vector3d camera_local = Eigen::Vector3d(0, 1.2, -2);

	// 	// Eigen::Vector3d root = rootpos.segment<3>(0);
	// 	// root = DPhy::projectToXZ(root);

	// 	// Eigen::AngleAxisd root_aa(root.norm(), root.normalized());
	// 	// camera_local = root_aa*camera_local + rootpos.segment<3>(3);

	// 	com[1] = 0.8;

	// 	// mCamera->SetCamera(com, camera_local);
	// 	mCamera->SetCenter(com);
	// }

	mCamera->Apply();

	DrawGround();
	DrawSkeletons();
	GUI::DrawStringOnScreen(0.8, 0.9, std::to_string(mCurFrame)+" / "+std::to_string(mTotalFrame), true, Eigen::Vector3d::Zero());
	// GUI::DrawStringOnScreen(0.8, 0.9, std::to_string(mMC->mTiming[mCurFrame])+" / "+std::to_string(mCurFrame), true, Eigen::Vector3d::Zero());
	// else GUI::DrawStringOnScreen(0.8, 0.9, std::to_string(mCurFrame), true, Eigen::Vector3d::Zero());
}
void
SceneMotionWidget::
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
SceneMotionWidget::
timerEvent(QTimerEvent* event)
{
	if(mPlay && this->mCurFrame == this->mTotalFrame - 1) {
		Step();
		mCurFrame += 1;
	} else if(mPlay){
		mCurFrame += 1;
	}
	SetFrame(this->mCurFrame);
	update();

}
void
SceneMotionWidget::
toggleDrawReg() {
	mDrawReg = !mDrawReg;

}
void
SceneMotionWidget::
toggleDrawSim() {
	if(mRunSim)
		mDrawSim = !mDrawSim;

}
bool 
SceneMotionWidget::
eventFilter(QObject * obj, QEvent * event)
{

    if ( event->type() == QEvent::KeyPress ) {
    	if(((QKeyEvent*)event)->key() == Qt::Key_D) {
			std::cout << "Pivot on wait" << std::endl;
			mMC->SwitchController("Pivot", 0);
		} else if(((QKeyEvent*)event)->key() == Qt::Key_W) {
			std::cout << "Punch on wait" << std::endl;
			mMC->SwitchController("Punch");
		} else if(((QKeyEvent*)event)->key() == Qt::Key_A) {
			std::cout << "Kick on wait" << std::endl;
			mMC->SwitchController("Kick", 5);
		} else if(((QKeyEvent*)event)->key() == Qt::Key_S) {
			std::cout << "Dodge on wait" << std::endl;
			mMC->SwitchController("Dodge");
		}

		if(((QKeyEvent*)event)->key() == Qt::Key_1) {
			std::cout << "Add new hitpoint" << std::endl;
			mMC->AddNewRandomHitPoint();
		} else if(((QKeyEvent*)event)->key() == Qt::Key_2) {
			std::cout << "clear and add" << std::endl;
			mMC->ClearHitPoints();
			mMC->AddNewRandomHitPoint();
		} else {
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		    pressedKeys.push_back(std::pair<int, std::chrono::steady_clock::time_point>(((QKeyEvent*)event)->key(), begin));
		}
    }
    else if ( event->type() == QEvent::KeyRelease )
    {
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::string actionType = mMC->GetNextAction();
		Eigen::VectorXd params;
		if(actionType=="Pivot" || actionType=="Dodge") {
			params.resize(1);
		} else if(actionType == "Kick" || actionType == "Punch") {
			params.resize(2);
		}
		params.setZero();
		for(int i = 0; i < pressedKeys.size(); i++) {
			if(pressedKeys[i].first == Qt::Key_W) {
				std::chrono::steady_clock::time_point begin = pressedKeys[i].second;
				double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
			}
			if(pressedKeys[i].first == Qt::Key_Left) {
				if(actionType=="Dodge") {
					params(0) = 0.9;
				}
				if(actionType=="Kick") {
					params(0) = 0.2;
				}
			}
			if(pressedKeys[i].first == Qt::Key_Down) {
				if(actionType=="Pivot") {
					if(params(0) != 0)
						params(0) = 0.3;
					else
						params(0) = 0.6;
				}
				if(actionType=="Dodge") {
					params(0) = 0.5;
				}
				if(actionType=="Kick") {
					params(0) = 0.5;
				}
				if(actionType=="Punch") {
					//force
					params(0) = 0.8;
				}
			}
			if(pressedKeys[i].first == Qt::Key_Right) {
				if(actionType=="Pivot") {
					if(params(0) != 0)
						params(0) = 0.3;
					else
						params(0) = 0.1;
				}
				if(actionType=="Dodge") {
					params(0) = 0.1;
				}
				if(actionType=="Kick") {
					params(0) = 0.7;
				}
			}
			if(pressedKeys[i].first == Qt::Key_Up) {
				if(actionType=="Kick") {
					params(1) = 0.6;
				}
				if(actionType=="Punch") {
					//distance
					params(1) = 0.8;
				}
			}
		}
		if(pressedKeys.size() != 0) {
			mMC->SetAction(params);
			pressedKeys.clear();
		}
    }


    return false;
}
void
SceneMotionWidget::
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
SceneMotionWidget::
mousePressEvent(QMouseEvent* event)
{
	mIsDrag = true;
	mButton = event->button();
	mPrevX = event->x();
	mPrevY = event->y();
}
void
SceneMotionWidget::
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
SceneMotionWidget::
mouseReleaseEvent(QMouseEvent* event)
{
	mIsDrag = false;
	mButton = Qt::NoButton;
	update();
}
void
SceneMotionWidget::
wheelEvent(QWheelEvent *event)
{
	if(event->angleDelta().y()>0)
	mCamera->Pan(0,-5,0,0);
	else
	mCamera->Pan(0,5,0,0);
	update();
}

void
SceneMotionWidget::
NextFrame()
{ 
	if(!mPlay) {
		this->mCurFrame += 1;
		this->SetFrame(this->mCurFrame);
	}
}
void
SceneMotionWidget::
PrevFrame()
{
	if(!mPlay && mCurFrame > 0) {
		this->mCurFrame -= 1;
		this->SetFrame(this->mCurFrame);
	}
}
void
SceneMotionWidget::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);
}
void 
SceneMotionWidget::
togglePlay() {
	mPlay = !mPlay;
}
void 
SceneMotionWidget::
Save() {
	time_t t;
	time(&t);

	std::string time_str = std::ctime(&t);

	// mController->SaveDisplayedData(mPath + "record_" + time_str, true);
}
