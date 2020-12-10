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
#include "RegressionMemory.h"
#include "GLfunctions.h"
#include "DART_interface.h"
#include "Controller.h"
#include <sys/socket.h>
#include "unistd.h"
#include <netinet/in.h>
#include <errno.h>
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
	void UEconnect();
	void UEclose();
	void UpdateParam(const bool& pressed);
	void UpdateRandomParam(const bool& pressed);
	void UpdatePrevParam(const bool& pressed);
	void UpdateNextParam(const bool& pressed);
	void updateIthParam(int i);
	void ResetController();
	void setValue(const int &x);
	void toggleDrawBvh();
	void toggleDrawSim();
	void toggleDrawReg();
	void toggleDrawExp();



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

	void connectionOpen();
	void connectionClose();
	int getCharacterTransformsForUE(char *buffer, int n);
	void sendMotion(int n);

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
	std::vector<Eigen::VectorXd> 	mMotion_exp;
	std::vector<Eigen::VectorXd> 	mMotion_obj;
	std::vector<Eigen::VectorXd> 	mMotion_points;
	std::vector<double>				mTiming; // Controller->GetCurrentLength()



	dart::dynamics::SkeletonPtr 	mSkel_bvh;
	dart::dynamics::SkeletonPtr 	mSkel_reg;
	dart::dynamics::SkeletonPtr 	mSkel_sim;
	dart::dynamics::SkeletonPtr 	mSkel_exp;
	dart::dynamics::SkeletonPtr 	mSkel_obj= nullptr;

	bool							mTrackCamera;

	bool 							mRunSim;
	bool 							mRunReg;
	bool 							mDrawBvh;
	bool 							mDrawSim;
	bool 							mDrawReg;
	bool 							mDrawExp;


	p::object 						mRegression;
	p::object 						mPPO;
	DPhy::ReferenceManager*			mReferenceManager;
	DPhy::Controller* 				mController;
	DPhy::RegressionMemory* 		mRegressionMemory;

	Eigen::VectorXd v_param;
	std::pair<Eigen::VectorXd, Eigen::VectorXd> mParamRange;

	Eigen::Vector3d 				mPoints;
	Eigen::Vector3d 				mPoints_exp;

	int regMemShow_idx = 0;

	std::vector<Eigen::VectorXd> mPoseRecords, mRefPoseRecords;

	// for socket network
	std::vector<std::string> mJointsUEOrder;
	std::vector<std::string> mObjectsUEOrder;

	bool							mIsConnected;
	int 							sockfd,clientfd;
	struct sockaddr_in serveraddr, clientaddr;
	char *mBuffer;
	char *mBuffer2;
};
#endif
