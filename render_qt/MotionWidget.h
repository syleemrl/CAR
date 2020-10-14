#ifndef __MOTION_WIDGET_H__
#define __MOTION_WIDGET_H__
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
#include "GLfunctions.h"
#include "DART_interface.h"
#include "Controller.h"
#pragma pop_macro("slots")
namespace p = boost::python;
namespace np = boost::python::numpy;
class MotionWidget : public QOpenGLWidget
{
	Q_OBJECT

public:
	MotionWidget();
	MotionWidget(std::string motion, std::string ppo, std::string reg);
	void UpdateMotion(std::vector<Eigen::VectorXd> motion, int type);

	void togglePlay();
	void RunPPO();
public slots:
	void NextFrame();
	void PrevFrame();
	void Reset();
	void UpdateParam(const bool& pressed);

	void setValueX(const int &x);
	void setValueY(const int &y);
	void toggleDrawBvh();
	void toggleDrawSim();
	void toggleDrawReg();

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

	std::vector<Eigen::VectorXd> 	mMotion_bvh;
	std::vector<Eigen::VectorXd> 	mMotion_reg;
	std::vector<Eigen::VectorXd> 	mMotion_sim;



	dart::dynamics::SkeletonPtr 	mSkel_bvh;
	dart::dynamics::SkeletonPtr 	mSkel_reg;
	dart::dynamics::SkeletonPtr 	mSkel_sim;
	bool							mTrackCamera;

	bool 							mRunSim;
	bool 							mRunReg;
	bool 							mDrawBvh;
	bool 							mDrawSim;
	bool 							mDrawReg;


	p::object 						mRegression;
	p::object 						mPPO;
	DPhy::ReferenceManager*			mReferenceManager;
	DPhy::Controller* 				mController;

	Eigen::VectorXd v_param;
	std::vector<Eigen::VectorXd> mParamRange;

};
#endif
