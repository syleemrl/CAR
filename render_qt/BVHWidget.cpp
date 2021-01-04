#include "BVHWidget.h"
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
BVHWidget::
BVHWidget()
  :mCamera(new Camera(1000, 650)),mCurFrame(0),mPlay(false),mTrackCamera(false)
{
	this->startTimer(30);
}
BVHWidget::
BVHWidget(std::vector<std::string> motion)
  :BVHWidget()
{
	mCurFrame = 0;

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

	for(int i =0 ; i < motion.size(); i++) {
    	mSkels_bvh.push_back(DPhy::SkeletonBuilder::BuildFromFile(path).first);
    	if(i == 0) {
       		DPhy::Character* ref = new DPhy::Character(path);
   			mReferenceManager = new DPhy::ReferenceManager(ref);
    	}
	  	mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion[i]);
	
	    std::vector<Eigen::VectorXd> p;
	    for(int j = 0; j < 1000; j++) {
	        Eigen::VectorXd p_frame = mReferenceManager->GetPosition(j, false);
	        p_frame(3) = 0;
	        //p_frame(3) = 1.5 * i - 0.75; 
	        p.push_back(p_frame);
	    }
	    mMotions_bvh.push_back(p);
	}
	mTotalFrame = 1000;
	
	DPhy::SetSkeletonColor(mSkels_bvh[0], Eigen::Vector4d(255./255., 102./255., 46./255., 1.0));
	//DPhy::SetSkeletonColor(mSkels_bvh[0], Eigen::Vector4d(148./255., 202./255., 53./255., 1.0));
	//DPhy::SetSkeletonColor(mSkels_bvh[0], Eigen::Vector4d(36./255., 162./255., 255./255., 1.0));

}
void
BVHWidget::
initializeGL()
{
	// shaderProgram.addShaderFromSourceFile(QGLShader::Vertex, QString::fromStdString(std::string(CAR_DIR)+std::string("/render_qt/vertexshader.txt")));
	// shaderProgram.addShaderFromSourceFile(QGLShader::Fragment, QString::fromStdString(std::string(CAR_DIR)+std::string("/render_qt/fragshader.txt")));
	// shaderProgram.link();
	// shaderProgram.bind();

	glClearColor(1,1,1,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	mCamera->Apply();
}
void
BVHWidget::
resizeGL(int w,int h)
{
	glViewport(0, 0, w, h);
	mCamera->SetSize(w, h);
	mCamera->Apply();
}
void
BVHWidget::
SetFrame(int n)
{
	for(int i = 0; i < mMotions_bvh.size(); i++)
    	mSkels_bvh[i]->setPositions(mMotions_bvh[i][n]);

}
void
BVHWidget::
DrawSkeletons()
{
	for(int i = 0; i < mSkels_bvh.size(); i++)
		GUI::DrawSkeleton(this->mSkels_bvh[i], 0);
	

}	
void
BVHWidget::
DrawGround()
{
	Eigen::Vector3d com_root;
	com_root = this->mSkels_bvh[0]->getRootBodyNode()->getCOM();

	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
BVHWidget::
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

}
void
BVHWidget::
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
	GLfloat fogColor[] = {0.9,0.9,0.9,1};
	glFogfv(GL_FOG_COLOR,fogColor);
	glFogi(GL_FOG_MODE,GL_LINEAR);
	// glFogf(GL_FOG_DENSITY,0.05);
	glFogf(GL_FOG_START,10.0);
	glFogf(GL_FOG_END,40.0);
}
void
BVHWidget::
timerEvent(QTimerEvent* event)
{
	if(mPlay && mCurFrame < mTotalFrame) {
		mCurFrame += 1;
	} 
	SetFrame(this->mCurFrame);
	update();
}
void
BVHWidget::
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
BVHWidget::
mousePressEvent(QMouseEvent* event)
{
	mIsDrag = true;
	mButton = event->button();
	mPrevX = event->x();
	mPrevY = event->y();
}
void
BVHWidget::
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
BVHWidget::
mouseReleaseEvent(QMouseEvent* event)
{
	mIsDrag = false;
	mButton = Qt::NoButton;
	update();
}
void
BVHWidget::
wheelEvent(QWheelEvent *event)
{
	if(event->angleDelta().y()>0)
	mCamera->Pan(0,-5,0,0);
	else
	mCamera->Pan(0,5,0,0);
	update();
}
void
BVHWidget::
NextFrame()
{ 
	if(!mPlay) {
		this->mCurFrame += 1;
		this->SetFrame(this->mCurFrame);
		std::cout << mCurFrame << std::endl;

	}
}
void
BVHWidget::
PrevFrame()
{
	if(!mPlay && mCurFrame > 0) {
		this->mCurFrame -= 1;
		this->SetFrame(this->mCurFrame);
		std::cout << mCurFrame << std::endl;

	}
}
void
BVHWidget::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);
}
void 
BVHWidget::
togglePlay() {
	mPlay = !mPlay;
}

