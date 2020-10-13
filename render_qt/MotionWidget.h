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
#include "GLfunctions.h"
#include "DART_interface.h"
#include "Character.h"
#include "Functions.h"
#pragma pop_macro("slots")
namespace p = boost::python;
namespace np = boost::python::numpy;
class MotionWidget : public QOpenGLWidget
{
	Q_OBJECT

public:
	MotionWidget();
	MotionWidget(dart::dynamics::SkeletonPtr skel_bvh, dart::dynamics::SkeletonPtr skel_reg, dart::dynamics::SkeletonPtr skel_sim);
	void UpdateMotion(std::vector<Eigen::VectorXd> motion, int type);

	void togglePlay();

public slots:
	void NextFrame();
	void PrevFrame();
	void Reset();
	
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

	Camera* 					mCamera;
	int								mPrevX,mPrevY;
	Qt::MouseButton 				mButton;
	bool 							mIsDrag;
	
	bool 							mPlay;
	int 							mCurFrame;

	std::vector<Eigen::VectorXd> 	mMotion_bvh;
	std::vector<Eigen::VectorXd> 	mMotion_reg;
	std::vector<Eigen::VectorXd> 	mMotion_sim;



	dart::dynamics::SkeletonPtr 	mSkel_bvh;
	dart::dynamics::SkeletonPtr 	mSkel_reg;
	dart::dynamics::SkeletonPtr 	mSkel_sim;
	bool							mTrackCamera;

	bool mMotionLoaded_bvh;
	bool mMotionLoaded_reg;
	bool mMotionLoaded_sim;

};
#endif
