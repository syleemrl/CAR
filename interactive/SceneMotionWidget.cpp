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
  mTrackCamera(false),  mDrawSim(true), mDrawReg(true), mRD(), mMT(mRD()), mUniform(0.0, 1.0)
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
	if(event->key() == Qt::Key_S) {
		std::cout << "S pressed" << std::endl;
		mMC->SwitchController("Idle");
	}
	if(event->key() == Qt::Key_D) {
		std::cout << "D pressed" << std::endl;
		mMC->SwitchController("Pivot");
	}
	if(event->key() == Qt::Key_W) {
		std::cout << "W pressed" << std::endl;
		mMC->SwitchController("Punch");
	}
	if(event->key() == Qt::Key_E) {
		std::cout << "E pressed" << std::endl;
		mMC->SwitchController("Kick");
	}
	if(event->key() == Qt::Key_A) {
		std::cout << "A pressed" << std::endl;
		mMC->SwitchController("Block");
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
