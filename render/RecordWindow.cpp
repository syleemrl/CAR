#include <GL/glew.h>
#include "RecordWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Functions.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;

RecordWindow::
RecordWindow(std::vector<std::string> motion)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false), mTimeStep(1 / 30.0)
{
	for(int i = 0; i < motion.size(); i++) {
		mDrawRef.push_back(true);
	}
	this->mTotalFrame = 0;
	this->num = motion.size();

	std::string skel_path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
	for(int i = 0; i < motion.size(); i++) {
		this->mRef.push_back(new DPhy::Character(skel_path));


		std::vector<Eigen::VectorXd> record_pos;
		record_pos.clear();

		int dof = this->mRef[i]->GetSkeleton()->getPositions().rows();
		std::string record_path = motion[i];
		std::ifstream is(record_path);
		
		char buffer[256];
		// is >> buffer;
		// int length = atoi(buffer);
		// if(this->mTotalFrame == 0 || length < mTotalFrame) {
		// 	mTotalFrame = length;
		// }

	//	for(int k = 0; k < length; k++) {
		int length = 0;
		while(!is.eof()) {
			Eigen::VectorXd p(dof);
			for(int j = 0; j < dof; j++) 
			{
				is >> buffer;
				p[j] = atof(buffer);
			}
			is >> buffer;
			is >> buffer;

			Eigen::Vector3d o;
			if(length == 0) {
				o = p.segment<3>(3);
				o[1] = 0;
			} else 
				o = mMemoryObj.back();

			// for(int j = 0; j < 3; j++) 
			// {
			// 	is >> buffer;
			// 	o(j) = atof(buffer);
			// }
			record_pos.push_back(p);
			mMemoryObj.push_back(o);
			length++;
		}
		record_pos.pop_back();
		mMemoryObj.pop_back();

		length -= 1;
	//	}

		is.close();
		if(this->mTotalFrame == 0 || length < mTotalFrame) {
			mTotalFrame = length;
		}
		mMemoryRef.push_back(record_pos);
		if(i == motion.size()-1)
			DPhy::SetSkeletonColor(this->mRef.back()->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));
		else
			DPhy::SetSkeletonColor(this->mRef.back()->GetSkeleton(), Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));

	}

	Eigen::Vector3d ori_otho(-1.0, 0, 0);
	for(int i = 1; i < motion.size(); i++) {
		for(int j = 0; j < mMemoryRef[i].size(); j++) {
			mMemoryRef[i][j].segment<3>(3) += ori_otho * i;
		}
	}
	this->mCurFrame = 0;
	this->mDisplayTimeout = 33;

	this->SetFrame(this->mCurFrame);

}
void
RecordWindow::
SetFrame(int n)
{
	if( n < 0 || n >= this->mTotalFrame )
	{
	 	std::cout << "Frame exceeds limits" << std::endl;
	 	return;
	}

	for(int i = 0; i < mRef.size(); i++) {
    	mRef[i]->GetSkeleton()->setPositions(mMemoryRef[i][n]);
    }
    mObj = mMemoryObj[0];

}

void
RecordWindow::
NextFrame()
{ 
	this->mCurFrame+=1;
	if (this->mCurFrame >= this->mTotalFrame) {
        this->mCurFrame = 0;
    }
	this->SetFrame(this->mCurFrame);
}
void
RecordWindow::
PrevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) {
        this->mCurFrame = this->mTotalFrame - 1;
    }
	this->SetFrame(this->mCurFrame);
}
void
RecordWindow::
DrawSkeletons()
{
	for(int i = 0; i < mRef.size(); i++) {
		if(this->mDrawRef[i]) {
			GUI::DrawSkeleton(this->mRef[i]->GetSkeleton(), 0);
		}
    }
}
void
RecordWindow::
DrawGround()
{
	GUI::DrawPoint(mObj, Eigen::Vector3d(1.0, 1.0, 1.0), 10);

	Eigen::Vector3d com_root;
	com_root = this->mRef[num / 2]->GetSkeleton()->getRootBodyNode()->getCOM();
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
RecordWindow::
Display() 
{

	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	dart::dynamics::SkeletonPtr skel = this->mRef[num / 2]->GetSkeleton();
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
RecordWindow::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);

}
void
RecordWindow::
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
		case '3': mDrawRef[2] = !mDrawRef[2];break;
		case '2': mDrawRef[1] = !mDrawRef[1];break;
		case '1': mDrawRef[0] = !mDrawRef[0];break;
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
RecordWindow::
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
RecordWindow::
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
RecordWindow::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}

void 
RecordWindow::
Step()
{	
	this->mCurFrame++;
	this->SetFrame(this->mCurFrame);
}
void
RecordWindow::
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