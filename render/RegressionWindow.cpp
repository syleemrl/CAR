#include <GL/glew.h>
#include "RegressionWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Functions.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;

RegressionWindow::
RegressionWindow(std::string motion, std::string network)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false), mTimeStep(1 / 30.0)
{
	this->mTotalFrame = 0;

	std::string skel_path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
	for(int i = 0; i < 5; i++) {
		this->mRef.push_back(new DPhy::Character(skel_path));
		std::vector<Eigen::VectorXd> memory;
		this->mMemoryRef.push_back(memory);
		DPhy::SetSkeletonColor(this->mRef[i]->GetSkeleton(), Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));

	}
	this->mRef_BVH = new DPhy::Character(skel_path);

	int dof = this->mRef[0]->GetSkeleton()->getPositions().rows();

	DPhy::ReferenceManager* referenceManager = new DPhy::ReferenceManager(this->mRef_BVH);
	referenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);

	std::vector<double> knots;
	knots.push_back(0);
	knots.push_back(12);
	knots.push_back(29);
	knots.push_back(37);
	knots.push_back(44);
	knots.push_back(52);
	knots.push_back(56);
	knots.push_back(59);	
	knots.push_back(64);
	knots.push_back(76);

	DPhy::MultilevelSpline* s = new DPhy::MultilevelSpline(1, referenceManager->GetPhaseLength());
	s->SetKnots(0, knots);
	
	std::vector<Eigen::VectorXd> cps;
	for(int i = 0; i < knots.size() + 3 ; i++) {
		cps.push_back(Eigen::VectorXd::Zero(dof));
	}

	Py_Initialize();
	np::initialize();
	try {
		p::object ppo_main = p::import("ppo");
		this->mPPO = ppo_main.attr("PPO")();
		std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + network;
		this->mPPO.attr("initRegression")(path);
	} catch (const p::error_already_set&) {
		PyErr_Print();
	}

	for(int i = 115, c = 0; i <= 150; i += 7, c++) {
		for(int j = 0; j < cps.size(); j++) {
			Eigen::VectorXd input(2);
			input << j, i / 100.0;
			p::object a = this->mPPO.attr("runRegression")(DPhy::toNumPyArray(input));
			np::ndarray na = np::from_object(a);
			cps[j] = DPhy::toEigenVector(na, dof);
		}
		s->SetControlPoints(0, cps);
		std::vector<Eigen::VectorXd> newpos;
		std::vector<Eigen::VectorXd> new_displacement = s->ConvertSplineToMotion();
		referenceManager->AddDisplacementToBVH(new_displacement, newpos);
		for(int j = 0; j < newpos.size(); j++) {
			newpos[j][3] += (c + 1);
		}
		for(int l = 0; l < 5; l++) {
			for(int j = 0; j < newpos.size(); j++) {
				mMemoryRef[c].push_back(newpos[j]);
			}
		}
	}

	mTotalFrame = 5 * referenceManager->GetPhaseLength();
	for(int l = 0; l < 5; l++) {
		for(int j = 0; j < referenceManager->GetPhaseLength(); j++) {
			mMemoryRefBVH.push_back(referenceManager->GetPosition(j));
		}
	}
	DPhy::SetSkeletonColor(this->mRef_BVH->GetSkeleton(), Eigen::Vector4d(235./255., 73./255., 73./255., 1.0));

	this->mCurFrame = 0;
	this->mDisplayTimeout = 33;

	this->SetFrame(this->mCurFrame);

}
void
RegressionWindow::
SetFrame(int n)
{
	if( n < 0 || n >= this->mTotalFrame )
	{
	 	std::cout << "Frame exceeds limits" << std::endl;
	 	return;
	}
	for(int i = 0; i < 5; i++)
    	mRef[i]->GetSkeleton()->setPositions(mMemoryRef[i][n]);
    mRef_BVH->GetSkeleton()->setPositions(mMemoryRefBVH[n]);

}

void
RegressionWindow::
NextFrame()
{ 
	this->mCurFrame+=1;
	if (this->mCurFrame >= this->mTotalFrame) {
        this->mCurFrame = 0;
    }
	this->SetFrame(this->mCurFrame);
}
void
RegressionWindow::
PrevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) {
        this->mCurFrame = this->mTotalFrame - 1;
    }
	this->SetFrame(this->mCurFrame);
}
void
RegressionWindow::
DrawSkeletons()
{
	for(int i = 0; i < 5; i++)
		GUI::DrawSkeleton(this->mRef[i]->GetSkeleton(), 0);
	GUI::DrawSkeleton(this->mRef_BVH->GetSkeleton(), 0);

}
void
RegressionWindow::
DrawGround()
{
	Eigen::Vector3d com_root;
	com_root = this->mRef[0]->GetSkeleton()->getRootBodyNode()->getCOM();
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
RegressionWindow::
Display() 
{

	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	dart::dynamics::SkeletonPtr skel = this->mRef[0]->GetSkeleton();
	Eigen::Vector3d com_root = skel->getRootBodyNode()->getCOM();
	Eigen::Vector3d com_front = skel->getRootBodyNode()->getTransform()*Eigen::Vector3d(0.0, 0.0, 2.0);

	if(this->mTrackCamera){
		Eigen::Vector3d com = skel->getRootBodyNode()->getCOM();
		Eigen::Isometry3d transform = skel->getRootBodyNode()->getTransform();
		com[1] = 0.8;

		Eigen::Vector3d camera_pos;
		camera_pos << -3, 1, 1.5;
		camera_pos = camera_pos + com;
		camera_pos[1] = 2;

		mCamera->SetCenter(com);
	}
	mCamera->Apply();

	glUseProgram(program);
	glPushMatrix();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glScalef(1.0, -1.0, 1.0);
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	// DrawSkeletons();
	glPopMatrix();
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	// glColor4f(0.7, 0.0, 0.0, 0.40);  /* 40% dark red floor color */
	DrawGround();
	DrawSkeletons();
	glDisable(GL_BLEND);

	glUseProgram(0);
	glutSwapBuffers();

}
void
RegressionWindow::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);

}
void
RegressionWindow::
Keyboard(unsigned char key,int x,int y) 
{
	switch(key)
	{
		case '`' :mIsRotate= !mIsRotate;break;
		case '[': mIsAuto=false;this->PrevFrame();break;
		case ']': mIsAuto=false;this->NextFrame();break;
		case 'o': this->mCurFrame-=99; this->PrevFrame();break;
		case 'p': this->mCurFrame+=99; this->NextFrame();break;
		case 's': std::cout << this->mCurFrame << std::endl;break;
		case 'r': Reset();break;
		case 't': mTrackCamera = !mTrackCamera; this->SetFrame(this->mCurFrame); break;
		case ' ':
			mIsAuto = !mIsAuto;
			break;
		case 27: exit(0);break;
		default : break;
	}
	// this->SetFrame(this->mCurFrame);

	// glutPostRedisplay();
}
void
RegressionWindow::
Mouse(int button, int state, int x, int y) 
{
	if(button == 3 || button == 4){
		if (button == 3)
		{
			mCamera->Pan(0,-5,0,0);
		}
		else
		{
			mCamera->Pan(0,5,0,0);
		}
	}
	else{
		if (state == GLUT_DOWN)
		{
			mIsDrag = true;
			mMouseType = button;
			mPrevX = x;
			mPrevY = y;
		}
		else
		{
			mIsDrag = false;
			mMouseType = 0;
		}
	}

	// glutPostRedisplay();
}
void
RegressionWindow::
Motion(int x, int y) 
{
	if (!mIsDrag)
		return;

	int mod = glutGetModifiers();
	if (mMouseType == GLUT_LEFT_BUTTON)
	{
		mCamera->Translate(x,y,mPrevX,mPrevY);
	}
	else if (mMouseType == GLUT_RIGHT_BUTTON)
	{
		mCamera->Rotate(x,y,mPrevX,mPrevY);
	

	}
	mPrevX = x;
	mPrevY = y;
}
void
RegressionWindow::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}

void 
RegressionWindow::
Step()
{	
	this->mCurFrame++;
	this->SetFrame(this->mCurFrame);
}
void
RegressionWindow::
Timer(int value) 
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	if( mIsAuto && this->mCurFrame == this->mTotalFrame - 1){
         Step();
	} else if( mIsAuto && this->mCurFrame < this->mTotalFrame - 1){
        this->mCurFrame++;
        SetFrame(this->mCurFrame);
        	
    }

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
	
	glutTimerFunc(std::max(0.0,mDisplayTimeout-elapsed), TimerEvent,1);
	glutPostRedisplay();

}