#include <GL/glew.h>
#include "SeqRecordWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Functions.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include "Functions.h"
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;

SeqRecordWindow::
SeqRecordWindow(std::vector<std::string> motion)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false), mTimeStep(1 / 30.0), mInterval(1), mDrawTarget(true)
{
	this->mTotalFrame = 0;
	this->mCurRecord = 0;
	std::string skel_path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
	this->mRef = new DPhy::Character(skel_path);
	auto skel = this->mRef->GetSkeleton();
	int length = 0;

	for(int i = 0; i < motion.size(); i++) {


		std::vector<Eigen::VectorXd> record_pos;
		record_pos.clear();

		int dof = skel->getPositions().rows();
		std::string record_path = motion[i];
		std::ifstream is(record_path);
		
		char buffer[256];
		int target_count = 1;
		while(!is.eof()) {
			Eigen::VectorXd p(dof);
			for(int j = 0; j < dof; j++) 
			{
				is >> buffer;
				p[j] = atof(buffer);
			}
				// is >> buffer;
				// is >> buffer;

			mMemoryRef.push_back(p);
			length++;
			if(target_count == 1) {
				Eigen::AngleAxisd aa(p.segment<3>(0).norm(), p.segment<3>(0).normalized());
				Eigen::Vector3d target = aa * Eigen::Vector3d(0.65, 0.43, 0.35) + p.segment<3>(3);
				mMemoryTarget.push_back(target);
			}
			if(target_count == 27) {
				skel->setPositions(p);
				skel->computeForwardKinematics(true,false,false);
				mMemoryKey.push_back(skel->getBodyNode("HandR")->getWorldTransform().translation());

			}
			target_count += 1;
		}
		mMemoryRef.pop_back();
		length -= 1;
	
		mEndKeyFrame.push_back(mMemoryRef.size() - 1);

		is.close();
	}
	if(this->mTotalFrame == 0 || length < mTotalFrame) {
		mTotalFrame = length;
	}
	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));

	this->mCurFrame = 0;
	this->mDisplayTimeout = 33;

	this->SetFrame(this->mCurFrame);

}
void
SeqRecordWindow::
SetFrame(int n)
{
	if( n < 0 || n >= this->mTotalFrame )
	{
	 	std::cout << "Frame exceeds limits" << std::endl;
	 	return;
	}

    mRef->GetSkeleton()->setPositions(mMemoryRef[n]);

}

void
SeqRecordWindow::
NextFrame()
{ 
	if (this->mCurFrame == this->mTotalFrame - 1) {
        this->mCurFrame = 0;
    } else if(mEndKeyFrame[mCurRecord] == mCurFrame) {
		mCurRecord += mInterval;
		if(mEndKeyFrame.size() > mCurRecord) {
			mCurFrame = mEndKeyFrame[mCurRecord - 1] + 1;
		} else {
			this->mCurFrame = 0;
		}
	} else {
		this->mCurFrame += 1;
	}
	this->SetFrame(this->mCurFrame);
}
void
SeqRecordWindow::
PrevFrame()
{
	if( this->mCurFrame == 0 ) {
        this->mCurFrame = this->mTotalFrame - 1;
    } else if(mEndKeyFrame[mCurRecord - 1] == mCurFrame - 1) {
		mCurRecord -= mInterval;
		if(0 <= mCurRecord) {
			mCurFrame = mEndKeyFrame[mCurRecord + 1] - 1;
		} else {
			this->mCurFrame = this->mTotalFrame - 1;
		}
	} else {
		this->mCurFrame -= 1;
	}
	this->SetFrame(this->mCurFrame);
}
void
SeqRecordWindow::
DrawSkeletons()
{
	GUI::DrawSkeleton(this->mRef->GetSkeleton(), 0);
	if(mDrawTarget) {
		std::vector<Eigen::Vector3d> keys;
		for(int i = 0; i <= mCurRecord; i += mInterval) {
			keys.push_back(mMemoryKey[i]);
		}
		GUI::DrawTrajectory(keys, keys.size(), Eigen::Vector3d(22./ 255., 194./ 255., 96./ 255.), false);
		GUI::DrawPoint(mMemoryTarget[mCurRecord], Eigen::Vector3d(194./255., 25./ 255., 48./ 255.), 10);

	}
}
void
SeqRecordWindow::
DrawGround()
{
	Eigen::Vector3d com_root;
	com_root = this->mRef->GetSkeleton()->getRootBodyNode()->getCOM();
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
SeqRecordWindow::
Display() 
{

	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	dart::dynamics::SkeletonPtr skel = this->mRef->GetSkeleton();
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
SeqRecordWindow::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);
	mCurRecord = 0;
	this->mMemoryKey.clear();
}
void
SeqRecordWindow::
Keyboard(unsigned char key,int x,int y) 
{
	switch(key)
	{
		case '`' :mIsRotate= !mIsRotate;break;
		case '[': mIsAuto=false;this->PrevFrame();break;
		case ']': mIsAuto=false;this->NextFrame();break;
		case 'o': this->mInterval -= 1; std::cout << "interval: " << this->mInterval << std::endl; break;
		case 'p': this->mInterval += 1; std::cout << "interval: " << this->mInterval << std::endl; break;
		case 's': std::cout << this->mCurFrame << std::endl;break;
		case 'r': Reset();break;
		case 't': mTrackCamera = !mTrackCamera; this->SetFrame(this->mCurFrame); break;
		case '1': mDrawTarget = !mDrawTarget;break;
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
SeqRecordWindow::
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
SeqRecordWindow::
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
SeqRecordWindow::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}
void
SeqRecordWindow::
Timer(int value) 
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	if( mIsAuto && this->mCurFrame < this->mTotalFrame - 1){
       NextFrame();
    }

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
	
	glutTimerFunc(std::max(0.0,mDisplayTimeout-elapsed), TimerEvent,1);
	glutPostRedisplay();

}