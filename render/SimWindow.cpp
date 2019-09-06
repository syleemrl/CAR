#include <GL/glew.h>
#include "SimWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Functions.h"
#include "matplotlibcpp.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;
namespace plt=matplotlibcpp;

SimWindow::
SimWindow(std::string motion, std::string network)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false), 
	mDrawOutput(true), mDrawRef(true), mDrawAdaptiveRef(true), mRunPPO(true), mTimeStep(1 / 30.0)
{
	if(network.compare("") == 0) {
		this->mDrawOutput = false;
		this->mDrawAdaptiveRef = false;
		this->mRunPPO = false;
	}

	this->mController = new DPhy::Controller(motion);
	this->mWorld = this->mController->GetWorld();

	std::string path = std::string(CAR_DIR) + std::string("/motion/") + motion + std::string(".bvh");
	this->mBVH = new DPhy::BVH();
	this->mBVH->Parse(path);

	path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

	this->mRef = new DPhy::Character(path);
	this->mRef->LoadBVHMap(path);
	this->mRef->ReadFramesFromBVH(this->mBVH);
	this->mRef->EditTrajectory(mBVH, 32, 4);

 	std::vector<double> x(this->mRef->GetMaxFrame()), y(this->mRef->GetMaxFrame());
 	for(int i = 0; i < this->mRef->GetMaxFrame(); i++) {
 		x.push_back(i);
 		y.push_back(this->mRef->GetTargetPositionsAndVelocitiesFromBVH(mBVH, i)->COMvelocity[1]);
 	}
 	plt::plot(x, y);
    plt::show();

	this->mAdaptiveRef = DPhy::SkeletonBuilder::BuildFromFile(path);

	DPhy::SetSkeletonColor(this->mController->GetSkeleton(), Eigen::Vector4d(0.73, 0.73, 0.73, 1.0));
	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));

	this->mSkelLength = 1;

	this->mController->Reset(false);
	DPhy::Frame* p_v_target = this->mRef->GetTargetPositionsAndVelocitiesFromBVH(mBVH, 0);
	mRef->GetSkeleton()->setPositions(p_v_target->position);
	mAdaptiveRef->setPositions(this->mController->GetAdaptivePosition());
	mRefContact = p_v_target->contact;

	if(this->mRunPPO)
	{
		Py_Initialize();
		np::initialize();
		try{
			p::object ppo_main = p::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			this->mPPO.attr("initRun")(network,
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
	this->Save();
	this->SetFrame(this->mCurFrame);

}
void 
SimWindow::
MemoryClear() {
    mMemory.clear();
    mMemoryRef.clear();
    mMemoryAdaptiveRef.clear();
    mMemoryRefContact.clear();
    mReward.clear();
}
void 
SimWindow::
Save() {
    SkeletonPtr humanoidSkel = this->mController->GetSkeleton();
    mMemory.emplace_back(humanoidSkel->getPositions());
    mMemoryRef.emplace_back(mRef->GetSkeleton()->getPositions());
    mMemoryAdaptiveRef.emplace_back(mAdaptiveRef->getPositions());
    mMemoryRefContact.emplace_back(mRefContact);
    this->mTotalFrame++;
    if(this->mRunPPO && !this->mController->IsTerminalState())
    {
    	if(this->mTotalFrame != 1) {
    		mReward = this->mController->GetRewardByParts();
    		mRewardTotal += mReward[0];

    	}
    	// std::cout << this->mTotalFrame-1 << ":";
    	// for(int i = 0; i < mReward.size(); i++) {
    	//  	std::cout << " " << mReward[0];
    	// }
    	// std::cout << std::endl;
    	std::cout << this->mTotalFrame-1 << ":" << mRewardTotal << std::endl;
	}

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

	 if (mMemory.size() <= n){
	     return;
	 }

    SkeletonPtr humanoidSkel = this->mController->GetSkeleton();
    humanoidSkel->setPositions(mMemory[n]);
    mRef->GetSkeleton()->setPositions(mMemoryRef[n]);
    mAdaptiveRef->setPositions(mMemoryAdaptiveRef[n]);
    mRefContact = mMemoryRefContact[n];
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
		GUI::DrawSkeleton(this->mController->GetSkeleton(), 0);
		for(int i = 0; i < this->mRefContact.size(); i++) {
			if(this->mController->CheckCollisionWithGround(this->mController->GetContactNodeName(i)))
				GUI::DrawBodyNode(this->mController->GetSkeleton(), Eigen::Vector4d(0.73*0.4, 0.73*0.4, 0.73*0.4, 1.0), this->mController->GetContactNodeName(i), 0);
		}
	}
	if(this->mDrawRef) {
		GUI::DrawSkeleton(this->mRef->GetSkeleton(), 0);
		for(int i = 0; i < this->mRefContact.size(); i++) {
			if(this->mRefContact[i] == 1)
				GUI::DrawBodyNode(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255.*0.4, 87./255.*0.4, 87./255.*0.4, 1.0), this->mRef->GetContactNodeName(i), 0);
		}
	}
	if(this->mDrawAdaptiveRef) {
		GUI::DrawSkeleton(this->mAdaptiveRef, 0);
	}
}
void
SimWindow::
DrawGround()
{
	Eigen::Vector3d com_root;
	if(this->mDrawOutput)
		com_root = this->mController->GetSkeleton()->getRootBodyNode()->getCOM();
	else 
		com_root = this->mRef->GetSkeleton()->getRootBodyNode()->getCOM();

	double ground_height = this->mController->GetSkeleton()->getRootBodyNode()->getCOM()[1]-0.5;
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
		skel = this->mController->GetSkeleton();
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
	double w = 1.05;
	this->mController->DeformCharacter(w);
	this->mSkelLength *= w;
	std::cout << this->mSkelLength << std::endl;

	std::vector<std::tuple<std::string, int, double>> deform;
	deform.push_back(std::make_tuple("FemurL", 1, w));
	deform.push_back(std::make_tuple("TibiaL", 1, w));
	deform.push_back(std::make_tuple("FemurR", 1, w));
	deform.push_back(std::make_tuple("TibiaR", 1, w));
		
	DPhy::SkeletonBuilder::DeformSkeleton(mRef->GetSkeleton(), deform);	
	this->mRef->RescaleOriginalBVH(w);
	
	this->mController->Reset(false);

	DPhy::Frame* p_v_target = this->mRef->GetTargetPositionsAndVelocitiesFromBVH(mBVH, 0);
	mRef->GetSkeleton()->setPositions(p_v_target->position);
	mAdaptiveRef->setPositions(this->mController->GetAdaptivePosition());
	mRefContact = p_v_target->contact;
	this->mRewardTotal = 0;
	this->mCurFrame = 0;
	this->mTotalFrame = 0;
	this->MemoryClear();
	this->Save();
	this->SetFrame(this->mCurFrame);

	DPhy::SetSkeletonColor(this->mController->GetSkeleton(), Eigen::Vector4d(0.73, 0.73, 0.73, 1.0));
	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));

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
		case 't': mTrackCamera = !mTrackCamera; this->SetFrame(this->mCurFrame); break;
		case '2': mDrawRef = !mDrawRef;break;
		case '3': if(this->mRunPPO) mDrawAdaptiveRef = !mDrawAdaptiveRef;break;
		case '1': if(this->mRunPPO) mDrawOutput = !mDrawOutput;break;
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
	if(this->mCurFrame < this->mRef->GetMaxFrame() - 1 || (this->mRunPPO && !mController->IsTerminalState())) 
	{
		if(this->mRunPPO)
		{
			auto state = this->mController->GetState();
			p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
			np::ndarray na = np::from_object(a);
			Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

			this->mController->SetAction(action);
			this->mController->Step();

		}
		DPhy::Frame* p_v_target = this->mRef->GetTargetPositionsAndVelocitiesFromBVH(mBVH, (this->mCurFrame+1));
		mRef->GetSkeleton()->setPositions(p_v_target->position);
		mAdaptiveRef->setPositions(this->mController->GetAdaptivePosition());
		mRefContact = p_v_target->contact;
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
        SetFrame(this->mCurFrame);
        	
    }

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
	
	glutTimerFunc(std::max(0.0,mDisplayTimeout-elapsed), TimerEvent,1);
	glutPostRedisplay();

}