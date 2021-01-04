#ifndef __BVH_WIDGET_H__
#define __BVH_WIDGET_H__
#include <vector>
#include <QOpenGLWidget>
#include <QTimerEvent>
#include <QGLShader>
#include <QKeyEvent>
#pragma push_macro("slots")
#undef slots
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "Camera.h"
#include "ReferenceManager.h"
#include "GLfunctions.h"
#include "DART_interface.h"
#pragma pop_macro("slots")
namespace p = boost::python;
namespace np = boost::python::numpy;
class BVHWidget : public QOpenGLWidget
{
	Q_OBJECT

public:
	BVHWidget();
	BVHWidget(std::vector<std::string> motion);
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

	Camera* 						mCamera;
	int								mPrevX,mPrevY;
	Qt::MouseButton 				mButton;
	bool 							mIsDrag;
	
	bool 							mPlay;
	int 							mCurFrame;
	int 							mTotalFrame;

	std::vector<std::vector<Eigen::VectorXd>> 	mMotions_bvh;
	std::vector<dart::dynamics::SkeletonPtr> 	mSkels_bvh;

	bool							mTrackCamera;


	DPhy::ReferenceManager*			mReferenceManager;

};
#endif
