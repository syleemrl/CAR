#include <GL/glew.h>
#include "SimWindowMulti.h"
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

SimWindowMulti::
SimWindowMulti(std::string motion, std::vector<std::string> network)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false), 
	 mDrawRef(true), mTimeStep(1 / 30.0)
{
	for(int i = 0; i < network.size(); i++)	
	{
		this->mController.emplace_back(new DPhy::Controller(motion, true));
		this->mDrawOutput.emplace_back(true);
	}

	std::string path = std::string(CAR_DIR) + std::string("/motion/") + motion + std::string(".bvh");
	this->mBVH = new DPhy::BVH();
	this->mBVH->Parse(path);

	path = std::string(CAR_DIR) + std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

	this->mRef = new DPhy::Character(path);
	this->mRef->LoadBVHMap(path);
	this->mRef->ReadFramesFromBVH(this->mBVH);

	double w0 = 1;
	std::vector<std::tuple<std::string, int, double>> deform;
	deform.push_back(std::make_tuple("ForeArmL", 0, w0));
	deform.push_back(std::make_tuple("ArmL", 0, w0));
	deform.push_back(std::make_tuple("ForeArmR", 0, w0));
	deform.push_back(std::make_tuple("ArmR", 0, w0));
	deform.push_back(std::make_tuple("FemurL", 1, w0));
	deform.push_back(std::make_tuple("TibiaL", 1, w0));
	deform.push_back(std::make_tuple("FemurR", 1, w0));
	deform.push_back(std::make_tuple("TibiaR", 1, w0));

	DPhy::SkeletonBuilder::DeformSkeletonLength(mRef->GetSkeleton(), deform);

	this->mRef->RescaleOriginalBVH(std::sqrt(w0));

	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));

	for(int i = 0; i < network.size(); i++)
	{
		DPhy::SetSkeletonColor(this->mController[i]->GetSkeleton(), Eigen::Vector4d((float) i / network.size(), 0.73, 0.73, 1.0));
		this->mController[i]->Reset(false);
	}
	DPhy::Frame* p_v_target = this->mRef->GetTargetPositionsAndVelocitiesFromBVH(mBVH, 0);
	mRef->GetSkeleton()->setPositions(p_v_target->position);

	Py_Initialize();
	np::initialize();
	try{
		for(int i = 0; i < network.size(); i++)
		{	
			p::object ppo_main = p::import("ppo");
			this->mPPO.emplace_back(ppo_main.attr("PPO")());
			this->mPPO[i].attr("initRun")(network[i],
									   this->mController[i]->GetNumState(), 
									   this->mController[i]->GetNumAction());
		}
	}
	catch (const p::error_already_set&)
	{
		PyErr_Print();
	}
	

	this->mCurFrame = 0;
	this->mTotalFrame = 0;
	this->mDisplayTimeout = 33;

	this->MemoryClear();
	this->Save();
	this->SetFrame(this->mCurFrame);
}
void 
SimWindowMulti::
MemoryClear() {
    mMemory.clear();
    mMemoryRef.clear();
}
void 
SimWindowMulti::
Save() {
	std::vector<Eigen::VectorXd> pos;
	for(int i = 0; i < mController.size(); i++)
	{
    	SkeletonPtr humanoidSkel = this->mController[i]->GetSkeleton();
    	pos.emplace_back(humanoidSkel->getPositions());
	}
	mMemory.emplace_back(pos);
    mMemoryRef.emplace_back(mRef->GetSkeleton()->getPositions());

    this->mTotalFrame++;
}

void
SimWindowMulti::
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
	for(int i = 0; i < mController.size(); i++)
	{
    	SkeletonPtr humanoidSkel = this->mController[i]->GetSkeleton();
    	humanoidSkel->setPositions(mMemory[n][i]);
	}

    mRef->GetSkeleton()->setPositions(mMemoryRef[n]);
}
void
SimWindowMulti::
NextFrame()
{ 
	this->mCurFrame+=1;
	if (this->mCurFrame >= this->mTotalFrame) {
        this->mCurFrame = 0;
    }
	this->SetFrame(this->mCurFrame);
}
void
SimWindowMulti::
PrevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) {
        this->mCurFrame = this->mTotalFrame - 1;
    }
	this->SetFrame(this->mCurFrame);
}
void
SimWindowMulti::
DrawSkeletons()
{
	for(int i =0 ; i < mController.size(); i++) 
	{
		if(this->mDrawOutput[i]) {
			GUI::DrawSkeleton(this->mController[i]->GetSkeleton(), 0);
		}
	}
	if(this->mDrawRef) {
		GUI::DrawSkeleton(this->mRef->GetSkeleton(), 0);
	}
}
void
SimWindowMulti::
DrawGround()
{
	Eigen::Vector3d com_root = this->mController[0]->GetSkeleton()->getRootBodyNode()->getCOM();
	double ground_height = this->mController[0]->GetSkeleton()->getRootBodyNode()->getCOM()[1]-0.5;
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
SimWindowMulti::
Display() 
{
	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	dart::dynamics::SkeletonPtr skel = this->mController[0]->GetSkeleton();
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
SimWindowMulti::
Reset()
{
	
	DPhy::Frame* p_v_target = this->mRef->GetTargetPositionsAndVelocitiesFromBVH(mBVH, 0);
	mRef->GetSkeleton()->setPositions(p_v_target->position);
	this->mCurFrame = 0;
	this->mTotalFrame = 0;
	this->MemoryClear();
	this->Save();
	this->SetFrame(this->mCurFrame);

	for(int i = 0; i < mController.size(); i++)
	{
		DPhy::SetSkeletonColor(this->mController[i]->GetSkeleton(), Eigen::Vector4d((float) i / mController.size(), 0.73, 0.73, 1.0));
		this->mController[i]->Reset(false);
	}	
	DPhy::SetSkeletonColor(this->mRef->GetSkeleton(), Eigen::Vector4d(235./255., 87./255., 87./255., 1.0));

}
void
SimWindowMulti::
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
		case '0': mDrawRef = !mDrawRef;break;
		case ' ':
			mIsAuto = !mIsAuto;
			break;
		case 27: exit(0);break;
		default : break;
	}
	if(key >= 49 && key <= 57) {
		int i = key - 49;
		if(i < mController.size())
			mDrawOutput[i] = !mDrawOutput[i];
	}
	// this->SetFrame(this->mCurFrame);

	// glutPostRedisplay();
}
void
SimWindowMulti::
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
SimWindowMulti::
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
SimWindowMulti::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}
std::string handle_pyerror()
{
    using namespace boost::python;
    using namespace boost;

    PyObject *exc,*val,*tb;
    object formatted_list, formatted;
    PyErr_Fetch(&exc,&val,&tb);
    handle<> hexc(exc),hval(allow_null(val)),htb(allow_null(tb)); 
    object traceback(import("traceback"));
    if (!tb) {
        object format_exception_only(traceback.attr("format_exception_only"));
        formatted_list = format_exception_only(hexc,hval);
    } else {
        object format_exception(traceback.attr("format_exception"));
        formatted_list = format_exception(hexc,hval,htb);
    }
    formatted = str("\n").join(formatted_list);
    return extract<std::string>(formatted);
}
void 
SimWindowMulti::
Step()
{
	if(this->mCurFrame < this->mRef->GetMaxFrame() - 1) 
	{
		
		for(int i = 0; i < mController.size(); i++)
		{
			try{
				auto state = this->mController[i]->GetState();
				p::object a = this->mPPO[i].attr("run")(DPhy::toNumPyArray(state));
				np::ndarray na = np::from_object(a);
				Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController[i]->GetNumAction());

				this->mController[i]->SetAction(action);
				this->mController[i]->Step();
			} catch(p::error_already_set) {
				std::string msg = handle_pyerror();
				std::cout << msg << std::endl;
				p::handle_exception();
				PyErr_Clear();
			}
		}
		DPhy::Frame* p_v_target = this->mRef->GetTargetPositionsAndVelocitiesFromBVH(mBVH, (this->mCurFrame+1));
		mRef->GetSkeleton()->setPositions(p_v_target->position);
		this->mCurFrame++;
		this->Save();

	}
	this->SetFrame(this->mCurFrame);
}
void
SimWindowMulti::
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