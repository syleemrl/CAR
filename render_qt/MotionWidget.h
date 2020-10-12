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
	MotionWidget(dart::dynamics::SkeletonPtr skel);
	// MotionWidget(p::object module,Preprocess* preprocess,const std::vector<Data*>& data,const std::vector<Category*>& category);
	void UpdateMotion(std::vector<Eigen::VectorXd> motion);

protected:
	void initializeGL() override;	
	void resizeGL(int w,int h) override;
	void paintGL() override;
	void initLights(double x, double z, double fx, double fz);

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

	std::vector<Eigen::VectorXd> 	mMotion;
	dart::dynamics::SkeletonPtr 	mSkel;
	bool							mTrackCamera;
	bool mMotionLoaded;
};
#endif
