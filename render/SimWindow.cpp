#include <GL/glew.h>
#include "SimWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Functions.h"
//#include "matplotlibcpp.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;
//namespace plt=matplotlibcpp;

SimWindow::
SimWindow(std::string motion, std::string network, std::string mode, std::string filename)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false), 
	mDrawOutput(true), mDrawRef(true), mDrawRef2(true), mRunPPO(true), mTimeStep(1 / 30.0),
	mWrap(false)
{
	if(network.compare("") == 0) {
		this->mDrawOutput = false;
		this->mRunPPO = false;
		this->mDrawRef2 = false;
	}
	this->filename = filename;
	this->mode = mode;

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

	this->mRef = new DPhy::Character(path);
	this->mRef2 = new DPhy::Character(path);

	this->mCharacter = new DPhy::Character(path);

	mReferenceManager = new DPhy::ReferenceManager(this->mRef);

	this->mController = new DPhy::Controller(std::string("/motion/") + motion, "", true);
	mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);
	mReferenceManager->GenerateMotionsFromSinglePhase(1000, false);
	//	mReferenceManager->EditMotion(1.5, "b");
	this->mWorld = this->mController->GetWorld();

	DPhy::SetSkeletonColor(this->mCharacter->GetSkeleton(), Eigen::Vector4d(0.73, 0.73, 0.73, 1.0));
	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));
	DPhy::SetSkeletonColor(this->mRef2->GetSkeleton(), Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));

	this->mSkelLength = 0.3;

	this->mController->Reset(false);
	DPhy::Motion* p_v_target = mReferenceManager->GetMotion(0);
	Eigen::VectorXd position = p_v_target->GetPosition();
	if(mWrap) {
		position.segment<6>(0).setZero();
		position[4] = 1.0;
	}	
	mRef->GetSkeleton()->setPositions(position);
	mRef2->GetSkeleton()->setPositions(position);
	mCharacter->GetSkeleton()->setPositions(position);

	if(this->mRunPPO)
	{
		Py_Initialize();
		np::initialize();
		try{
			p::object ppo_main = p::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			path = std::string(CAR_DIR)+ std::string("/network/output/") + network;
			this->mPPO.attr("initRun")(path,
									   this->mController->GetNumState(), 
									   this->mController->GetNumAction());

		}
		catch (const p::error_already_set&)
		{
			PyErr_Print();
		}
	}

	this->mCurFrame = 0;
	this->mTotalFrame = 0;
	this->mDisplayTimeout = 33;
	this->mRewardTotal = 0;

	this->MemoryClear();
	this->Save(this->mCurFrame);
	this->SetFrame(this->mCurFrame);
}
void 
SimWindow::
MemoryClear() {
    mMemory.clear();
    mMemoryRef.clear();
    mMemoryCOM.clear();
    mMemoryCOMRef.clear();
    mMemoryRef2.clear();
    mMemoryCOMRef2.clear();
    mMemoryGRF.clear();
    mMemoryFootContact.clear();
    mReward.clear();
}
void 
SimWindow::
Save(int n) {
	DPhy::Motion* p_v_target = mReferenceManager->GetMotion(n);
	Eigen::VectorXd position = p_v_target->GetPosition();
	if(mWrap) {
		position.segment<6>(0).setZero();
		position[4] = 1.0;
	}
	mRef->GetSkeleton()->setPositions(position);
    mMemoryRef.emplace_back(mRef->GetSkeleton()->getPositions());
    mMemoryCOMRef.emplace_back(mRef->GetSkeleton()->getCOM());
    this->mTotalFrame++;
    if(this->mRunPPO && n < this->mController->GetRecordSize())
    {
    	// if(this->mTotalFrame != 1) {
    	// 	mReward = this->mController->GetRewardByParts();
    	// 	mRewardTotal += mReward[0];
    	// 	mMemoryGRF.emplace_back(this->mController->GetGRF());

    	// }
    	position = this->mController->GetPositions(n);
		if(mWrap) {
			position.segment<6>(0).setZero();
			position[4] = 1.0;
		}		    	
		mMemory.emplace_back(position);
    	mMemoryCOM.emplace_back(this->mController->GetCOM(n));	
    	mMemoryFootContact.emplace_back(this->mController->GetFootContact(n));
    	p_v_target = mReferenceManager->GetMotion(this->mController->GetTime(n));
		mRef2->GetSkeleton()->setPositions(p_v_target->GetPosition());
   	 	mMemoryRef2.emplace_back(mRef2->GetSkeleton()->getPositions());
    	mMemoryCOMRef2.emplace_back(mRef2->GetSkeleton()->getCOM());

    	std::cout << this->mTotalFrame-1 << ":" << mRewardTotal << std::endl;
	}
}
void
SimWindow::
SaveReferenceData(std::string path) 
{
	std::string path_full = std::string(CAR_DIR) + std::string("/") + path  + std::string("_ref");
	std::cout << "save results to" << path_full << std::endl;

	std::ofstream ofs(path_full);

	ofs << mMemoryRef.size() << std::endl;
	for(auto t: mMemoryRef) {
		ofs << t.transpose() << std::endl;
	}
	std::cout << "saved position: " << mMemoryRef.size() << ", "<< mReferenceManager->GetPhaseLength() << ", " << mMemoryRef[0].rows() << std::endl;

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

	 if ((mRunPPO && mMemory.size() <= n) || mMemoryRef.size() <= n){
	     return;
	 }
  //  SkeletonPtr humanoidSkel = this->mController->GetSkeleton();
  	if(mRunPPO) 
  	{
  		mCharacter->GetSkeleton()->setPositions(mMemory[n]);
  		mFootContact = mMemoryFootContact[n];
  		mRef2->GetSkeleton()->setPositions(mMemoryRef2[n]);

  	}
    mRef->GetSkeleton()->setPositions(mMemoryRef[n]);
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
	if(this->mDrawOutput) {
		GUI::DrawSkeleton(this->mCharacter->GetSkeleton(), 0);
		GUI::DrawTrajectory(this->mMemoryCOM, this->mCurFrame, Eigen::Vector3d(0.9, 0.9, 0.9));
		if(this->mMemoryGRF.size() != 0) {
			std::vector<Eigen::VectorXd> grfs = this->mMemoryGRF.at(this->mCurFrame-1);
			GUI::DrawForces(grfs, Eigen::Vector3d(1, 0, 0));
		}
		GUI::DrawFootContact(this->mCharacter->GetSkeleton(), mFootContact);
	}
	if(this->mDrawRef) {
		GUI::DrawSkeleton(this->mRef->GetSkeleton(), 0);
		GUI::DrawTrajectory(this->mMemoryCOMRef, this->mCurFrame);
	}
	if(this->mDrawRef2) {
		GUI::DrawSkeleton(this->mRef2->GetSkeleton(), 0);
		GUI::DrawTrajectory(this->mMemoryCOMRef2, this->mCurFrame);
	}
}
void
SimWindow::
DrawGround()
{
	Eigen::Vector3d com_root;
	if(this->mDrawOutput)
		com_root = this->mCharacter->GetSkeleton()->getRootBodyNode()->getCOM();
	else 
		com_root = this->mRef->GetSkeleton()->getRootBodyNode()->getCOM();

	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
SimWindow::
Display() 
{
	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	dart::dynamics::SkeletonPtr skel;
	if(this->mDrawOutput)
		skel = this->mCharacter->GetSkeleton();
	else
		skel = this->mRef->GetSkeleton();
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
SimWindow::
Reset()
{
	
	this->mController->Reset(false);

	DPhy::Motion* p_v_target = mReferenceManager->GetMotion(0);
	Eigen::VectorXd position = p_v_target->GetPosition();
	if(mWrap) {
		position.segment<6>(0).setZero();
		position[4] = 1.0;
	}	
	mRef->GetSkeleton()->setPositions(position);
	mRef2->GetSkeleton()->setPositions(position);

	this->mRewardTotal = 0;
	this->mCurFrame = 0;
	this->mTotalFrame = 0;
	this->MemoryClear();
	this->Save(this->mCurFrame);
	this->SetFrame(this->mCurFrame);

	DPhy::SetSkeletonColor(this->mCharacter->GetSkeleton(), Eigen::Vector4d(0.73, 0.73, 0.73, 1.0));
	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));
	DPhy::SetSkeletonColor(this->mRef2->GetSkeleton(), Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));

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
		case 'r': Reset();break;
		case 'w': mWrap = !mWrap; break;
		case 't': mTrackCamera = !mTrackCamera; this->SetFrame(this->mCurFrame); break;
		case '3': if(this->mRunPPO) mDrawRef2 = !mDrawRef2;break;
		case '2': mDrawRef = !mDrawRef;break;
		case '1': if(this->mRunPPO) mDrawOutput = !mDrawOutput;break;
		case ' ':
			mIsAuto = !mIsAuto;
			break;
		case 'R': SaveReferenceData(filename); break;
		case 'D': this->mController->SaveDisplayedData(filename); break;
		case 'S': this->mController->SaveStats(filename); break;
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
	// if(this->mCurFrame < this->mRef->GetMaxFrame() - 1) 
	// {
	for(int i = 1; i <= 2; i++) {
		if(this->mRunPPO)
		{
			auto state = this->mController->GetState();
			p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
			np::ndarray na = np::from_object(a);
			Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

			this->mController->SetAction(action);
			this->mController->Step();

		}
	// }
	}		
	this->mCurFrame++;
	this->Save(this->mCurFrame);
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
        SetFrame(this->mCurFrame);
        	
    }

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
	
	glutTimerFunc(std::max(0.0,mDisplayTimeout-elapsed), TimerEvent,1);
	glutPostRedisplay();

}