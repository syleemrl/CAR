#include <GL/glew.h>
#include "SplineWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "Functions.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;

SplineWindow::
SplineWindow(std::string motion, std::string record, std::string record_type)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false), mTimeStep(1 / 30.0), mDrawRef2(false)
{
	this->mTotalFrame = 0;

	std::string skel_path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
	
	this->mRef = new DPhy::Character(skel_path);
	this->mRef3 = new DPhy::Character(skel_path); 

	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));
	DPhy::SetSkeletonColor(this->mRef3->GetSkeleton(), Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));

	if(record_type.compare("position") == 0) {
		mDrawRef2 = true;
		this->mRef2 = new DPhy::Character(skel_path); 

		DPhy::SetSkeletonColor(this->mRef2->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));
	}

	int dof = this->mRef->GetSkeleton()->getPositions().rows();

	DPhy::ReferenceManager* referenceManager = new DPhy::ReferenceManager(this->mRef);
	referenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);
	referenceManager->InitOptimization(1, "");

	std::vector<double> knots = referenceManager->GetKnots();

	DPhy::MultilevelSpline* s = new DPhy::MultilevelSpline(1, referenceManager->GetPhaseLength());
	s->SetKnots(0, knots);
	
	std::ifstream is(record);
		
	char buffer[256];

	int length = 0;
	double reward = 0;

	std::vector<Eigen::VectorXd> pos;
	std::vector<double> step;

	int n = 0;
	std::vector<Eigen::VectorXd> cps_mean;
	std::vector<Eigen::VectorXd> cps;

	for(int i = 0; i < knots.size() + 3 ; i++) {
		cps_mean.push_back(Eigen::VectorXd::Zero(dof));
	}

	Eigen::VectorXd targetBase = referenceManager->GetTargetBase();
	Eigen::VectorXd targetUnit = referenceManager->GetTargetUnit();
	Eigen::VectorXd targetIdx(targetBase.size());
	targetIdx(0) = 0;
	while(!is.eof()) {
		if(record_type.compare("spline") == 0) {
			// cps number
			is >> buffer;
			Eigen::VectorXd tp(targetBase.rows());
			Eigen::VectorXd idx(targetBase.rows());
			for(int j = 0; j < targetBase.rows(); j++) 
			{
				is >> buffer;
				tp[j] = atof(buffer);
				idx[j] = std::floor((tp[j] - targetBase[j]) / targetUnit[j]);
			}
			// comma
			is >> buffer;

			Eigen::VectorXd cp(dof);
			for(int j = 0; j < dof; j++) 
			{
				is >> buffer;
				cp[j] = atof(buffer);
			}
			// comma
			is >> buffer;
			// reward
			is >> buffer;
		
			cps.push_back(cp);

			if(cps.size() == knots.size() + 3) {
				s->SetControlPoints(0, cps);
				std::vector<Eigen::VectorXd> displacement = s->ConvertSplineToMotion();	
				std::vector<Eigen::VectorXd> new_pos;
				referenceManager->AddDisplacementToBVH(displacement, new_pos);

				for(int i = 0; i < new_pos.size(); i++) {
					length += 1;
					mMemoryRef.push_back(new_pos[i]);
				}	
				for(int i = 0; i < cps.size(); i++) {
					cps_mean[i] += cps[i];
				}

				n += 1;
				cps.clear();
			}
		} else if(record_type.compare("position") == 0) {
			Eigen::VectorXd p(dof);
			for(int j = 0; j < dof; j++) 
			{
				is >> buffer;
				p[j] = atof(buffer);
			}

			is >> buffer;
			double cur_step = atof(buffer);			
			is >> buffer;
			double cur_reward = atof(buffer);
			is >> buffer;
			double cur_reward2 = atof(buffer);
			if(reward == 0)
				reward = cur_reward;

			// next phase
			if(cur_reward != reward) {
				reward = cur_reward;
			
				std::vector<std::pair<Eigen::VectorXd,double>> displacement;
				pos = DPhy::Align(pos, referenceManager->GetPosition(std::fmod(step[0], referenceManager->GetPhaseLength())).segment<6>(0));
				std::vector<std::pair<Eigen::VectorXd,double>> trajectory;
				for(int i = 0; i < pos.size(); i++) {
					trajectory.push_back(std::pair<Eigen::VectorXd,double>(pos[i], step[i]));
				}
				referenceManager->GetDisplacementWithBVH(trajectory, displacement);
				s->ConvertMotionToSpline(displacement);

				std::vector<Eigen::VectorXd> new_displacement = s->ConvertSplineToMotion();
				std::vector<Eigen::VectorXd> new_pos;
				referenceManager->AddDisplacementToBVH(new_displacement, new_pos);

				cps = s->GetControlPoints(0);
				for(int i = 0; i < cps.size(); i++) {
					cps_mean[i] += cps[i];
				}
				n += 1;

				int l = std::min(pos.size(), new_pos.size());
				for(int i = 0; i < l; i++) {
					length += 1;
					mMemoryRef.push_back(pos[i]);
					new_pos[i][3] += 1.5;
					mMemoryRef2.push_back(new_pos[i]);
				}
				pos.clear();
				step.clear();
			}

			pos.push_back(p);
			step.push_back(cur_step);
		}
	}
	is.close();

	for(int i = 0; i < cps_mean.size(); i++) {
		cps_mean[i] /= n;
	}
	s->SetControlPoints(0, cps_mean);
	std::vector<Eigen::VectorXd> displacement = s->ConvertSplineToMotion();
	std::vector<Eigen::VectorXd> new_pos;
	referenceManager->AddDisplacementToBVH(displacement, new_pos);
	for(int i = 0; i < referenceManager->GetPhaseLength(); i++) {
		new_pos[i][3] += 3;
	}
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < referenceManager->GetPhaseLength(); j++) {
			mMemoryRef3.push_back(new_pos[j]);
		}
	}
	
	if(this->mTotalFrame == 0 || length < mTotalFrame) {
		mTotalFrame = length;
	}

	this->mCurFrame = 0;
	this->mDisplayTimeout = 33;

	this->SetFrame(this->mCurFrame);

}
void
SplineWindow::
SetFrame(int n)
{
	if( n < 0 || n >= this->mTotalFrame )
	{
	 	std::cout << "Frame exceeds limits" << std::endl;
	 	return;
	}

    mRef->GetSkeleton()->setPositions(mMemoryRef[n]);
    if(mDrawRef2)  {
    	mRef2->GetSkeleton()->setPositions(mMemoryRef2[n]);
    }
    mRef3->GetSkeleton()->setPositions(mMemoryRef3[n]);

}

void
SplineWindow::
NextFrame()
{ 
	this->mCurFrame+=1;
	if (this->mCurFrame >= this->mTotalFrame) {
        this->mCurFrame = 0;
    }
	this->SetFrame(this->mCurFrame);
}
void
SplineWindow::
PrevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) {
        this->mCurFrame = this->mTotalFrame - 1;
    }
	this->SetFrame(this->mCurFrame);
}
void
SplineWindow::
DrawSkeletons()
{
	GUI::DrawSkeleton(this->mRef->GetSkeleton(), 0);
	if(mDrawRef2) {
		GUI::DrawSkeleton(this->mRef2->GetSkeleton(), 0);
	}
	GUI::DrawSkeleton(this->mRef3->GetSkeleton(), 0);

}
void
SplineWindow::
DrawGround()
{
	Eigen::Vector3d com_root;
	com_root = this->mRef->GetSkeleton()->getRootBodyNode()->getCOM();
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
SplineWindow::
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
SplineWindow::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);

}
void
SplineWindow::
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
SplineWindow::
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
SplineWindow::
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
SplineWindow::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}

void 
SplineWindow::
Step()
{	
	this->mCurFrame++;
	this->SetFrame(this->mCurFrame);
}
void
SplineWindow::
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