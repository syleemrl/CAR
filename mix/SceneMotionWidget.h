#ifndef __SCENE_MOTION_WIDGET_H__
#define __SCENE_MOTION_WIDGET_H__
#include <vector>
#include <QOpenGLWidget>
#include <QTimerEvent>
#include <QKeyEvent>
#pragma push_macro("slots")
#undef slots
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "Camera.h"
#include "ReferenceManager.h"
#include "RegressionMemory.h"
#include "GLfunctions.h"
#include "DART_interface.h"
#include "Controller.h"
#include "MetaController.h"
#pragma pop_macro("slots")
namespace p = boost::python;
namespace np = boost::python::numpy;
class SceneMotionWidget : public QOpenGLWidget
{
	Q_OBJECT

public:
	SceneMotionWidget();
	SceneMotionWidget(std::string motion, std::string ppo, std::string reg);
	
	void togglePlay();
	void RunPPO();
public slots:
	void NextFrame();
	void PrevFrame();
	void Reset();
	void Save();
	
	void setValue(const int &x);
	void toggleDrawBvh();
	void toggleDrawSim();
	void toggleDrawReg();
	void followCamera();
	
protected:
	void initializeGL() override;	
	void resizeGL(int w,int h) override;
	void paintGL() override;
	void initLights();

	void timerEvent(QTimerEvent* event);
	void keyPressEvent(QKeyEvent *event);

	void mousePressEvent(QMouseEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void wheelEvent(QWheelEvent *event);

	void DrawGround();
	void DrawSkeletons();
	void SetFrame(int n);
 	void initNetworkSetting(std::string motion, std::string network);

	Camera* 						mCamera;
	int								mPrevX,mPrevY;
	Qt::MouseButton 				mButton;
	bool 							mIsDrag;
	
	bool 							mPlay;
	int 							mCurFrame;
	int 							mTotalFrame;
	std::string						mPath;

	std::vector<Eigen::VectorXd> 	mMotion_bvh;
	std::vector<Eigen::VectorXd> 	mMotion_reg;
	std::vector<Eigen::VectorXd> 	mMotion_sim;

	std::vector<double>				mTiming; // Controller->GetCurrentLength()

	dart::dynamics::SkeletonPtr 	mSkel_bvh;
	dart::dynamics::SkeletonPtr 	mSkel_reg;
	dart::dynamics::SkeletonPtr 	mSkel_sim;

	dart::dynamics::SkeletonPtr 	mSkel_obj;

	bool							mTrackCamera;

	bool 							mRunSim;
	bool 							mRunReg;

	bool 							mDrawBvh;
	bool 							mDrawSim;
	bool 							mDrawReg;

	// p::object 						mRegression;
	// p::object 						mPPO;
	// DPhy::ReferenceManager*			mReferenceManager;
	// DPhy::Controller* 				mController;
	// DPhy::RegressionMemory* 		mRegressionMemory;

	// Eigen::VectorXd v_param;
	// std::pair<Eigen::VectorXd, Eigen::VectorXd> mParamRange;

	// Eigen::Vector3d 				mPoints;
	// Eigen::Vector3d 				mPoints_exp;
	// int regMemShow_idx= 0;

	std::random_device mRD;
	std::mt19937 mMT;
	std::uniform_real_distribution<double> mUniform;

	DPhy::MetaController* mMC;

};
#endif