#include "MotionWidget.h"
#include <GL/glu.h>
#include <iostream>
#include <Eigen/Geometry>
#include <chrono>
MotionWidget::
MotionWidget()
  :mCamera(new Camera(50, 50)),mMotionLoaded(false),mCurFrame(0),mPlay(false),mTrackCamera(false)
{
	this->startTimer(30);
}
MotionWidget::
MotionWidget(dart::dynamics::SkeletonPtr skel)
  :MotionWidget()
{
	mSkel = skel;
	DPhy::SetSkeletonColor(mSkel, Eigen::Vector4d(235./255., 73./255., 73./255., 1.0));

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
	// mCamera->SetSize(w,h);

	// mCamera->SetCemera(w,h);
}
void
MotionWidget::
SetFrame(int n)
{
    mSkel->setPositions(mMotion[n]);

}
void
MotionWidget::
DrawSkeletons()
{
	GUI::DrawSkeleton(this->mSkel, 0);
}
void
MotionWidget::
DrawGround()
{
	Eigen::Vector3d com_root;
	com_root = this->mSkel->getRootBodyNode()->getCOM();
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
MotionWidget::
paintGL()
{
	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	initLights(0, 0, 0, 0);
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
initLights(double x, double z, double fx, double fz)
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
	if(mMotionLoaded) {
		mCurFrame+=1;
		if(mCurFrame >= mMotion.size()) {
			mCurFrame = 0;
		}
		SetFrame(this->mCurFrame);
	} else
		mCurFrame = 0;
	update();
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
	mCamera->Pan(0,5,0,0);
	else
	mCamera->Pan(0,-5,0,0);
	update();
}
void 
MotionWidget::
UpdateMotion(std::vector<Eigen::VectorXd> motion)
{
	mMotionLoaded = true;
	mMotion = motion;
}
