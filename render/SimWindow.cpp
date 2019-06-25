#include <GL/glew.h>
#include "SimWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Functions.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;

SimWindow::
SimWindow(std::string motion)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false)
{
	this->mController = new DPhy::Controller(motion);
	this->mWorld = this->mController->GetWorld();

	DPhy::SetSkeletonColor(this->mWorld->getSkeleton("Humanoid"), Eigen::Vector4d(0.73, 0.73, 0.73, 1.0));

	this->mController->Reset(false);

	this->mCurFrame = 0;
	this->mTotalFrame = 0;
	this->mDisplayTimeout = 33;

	this->MemoryClear();
	this->Save();
	this->SetFrame(this->mCurFrame);
}
void 
SimWindow::
MemoryClear() {
    mMemory1.clear();
}
void 
SimWindow::
Save() {
    SkeletonPtr humanoidSkel = this->mWorld->getSkeleton("Humanoid");
    mMemory1.emplace_back(humanoidSkel->getPositions());
 
    this->mTotalFrame++;
}

void
SimWindow::
SetFrame(int n)
{
	 if( n < 0 || n >= this->mTotalFrame )
	 {
	 	std::cout << "Frame exceeds limits" << std::endl;
	 	return;
	 }

	 if (mMemory1.size() <= n){
	     return;
	 }

	int dof_index = 0;
    SkeletonPtr humanoidSkel = this->mWorld->getSkeleton("Humanoid");
    humanoidSkel->setPositions(mMemory1[n]);

}
void
SimWindow::
NextFrame()
{ 
	this->mCurFrame+=1;
	if (this->mCurFrame >= this->mTotalFrame) {
        this->mCurFrame = 0;
    }
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
PrevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) {
        this->mCurFrame = this->mTotalFrame - 1;
    }
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
DrawSkeletons()
{
	GUI::DrawSkeleton(this->mWorld->getSkeleton("Humanoid"), 0);

}
void
SimWindow::
DrawGround()
{
	Eigen::Vector3d com_root = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
	double ground_height = this->mWorld->getSkeleton("Ground")->getRootBodyNode()->getCOM()[1]+0.5;
	GUI::DrawGround((int)com_root[0], (int)com_root[2], ground_height);
}
void
SimWindow::
Display() 
{
	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	Eigen::Vector3d com_root = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
	Eigen::Vector3d com_front = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getTransform()*Eigen::Vector3d(0.0, 0.0, 2.0);

	if(this->mTrackCamera){
		Eigen::Vector3d com = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
		Eigen::Isometry3d transform = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getTransform();
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
	DrawSkeletons();
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
SimWindow::
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
		case 'r': this->mCurFrame=0;this->SetFrame(this->mCurFrame);break;
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
SimWindow::
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
SimWindow::
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
SimWindow::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}

void 
SimWindow::
Step()
{
	if(this->mController->FollowBvh()) 
	{
		this->mCurFrame++;
		this->Save();
	}
	this->SetFrame(this->mCurFrame);
		
}
void
SimWindow::
Timer(int value) 
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	if( mIsAuto && this->mCurFrame == this->mTotalFrame - 1){
          Step();
	} else if( mIsAuto && this->mCurFrame < this->mTotalFrame - 1){
        this->mCurFrame++;
        // SetFrame(this->mCurFrame);
        	
    }

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	double elasped = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
	
	glutTimerFunc(std::max(0.0,mDisplayTimeout-elasped), TimerEvent,1);
	glutPostRedisplay();

}