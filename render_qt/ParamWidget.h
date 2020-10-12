// #ifndef __PARAM_WIDGET_H__
// #define __PARAM_WIDGET_H__
// #include <QOpenGLWidget>
// #include <QTimerEvent>
// #include <QKeyEvent>
// #pragma push_macro("slots")
// #undef slots
// #include <boost/python.hpp>
// #include <boost/python/numpy.hpp>
// #pragma pop_macro("slots")
// namespace p = boost::python;
// namespace np = boost::python::numpy;
// class ParamWidget : public QOpenGLWidget
// {
// 	Q_OBJECT

// public:
// 	MotionWidget();
// 	MotionWidget(p::object module,Preprocess* preprocess,const std::vector<Data*>& data,const std::vector<Category*>& category);

// signals:
// 	void curFrameChanged(int curFrame);
// 	void motionLoaded(bool flag);

// public slots:
// 	void setFrame(int curFrame);
// 	void drawMovement(const int& i);
// 	void togglePlay(const bool& pressed);

// protected:
// 	void initializeGL() override;	
// 	void resizeGL(int w,int h) override;
// 	void paintGL() override;
// 	void initLights();

// 	void timerEvent(QTimerEvent* event);
// 	void keyPressEvent(QKeyEvent *event);

// 	void mousePressEvent(QMouseEvent* event);
// 	void mouseMoveEvent(QMouseEvent* event);
// 	void mouseReleaseEvent(QMouseEvent* event);
// 	void wheelEvent(QWheelEvent *event);

// 	Camera* 						mCamera;
// 	int								mPrevX,mPrevY;
// 	Qt::MouseButton 				mButton;
// 	bool 							mIsDrag;
	
// 	bool 							mPlay;
// 	int 							mCurFrame;

// 	int 							mMotionIdx;

// 	Preprocess*						mPreprocess;
// 	std::vector<CyclizedMotion*>		mCyclizedMotion;

// 	Eigen::Vector3d					mGlobalTranslation;

// 	std::vector<Data*>				mData;
// 	std::vector<Category*>			mCategory;

// 	std::vector<CyclizedMotion*>	mEncodeData;

// 	Eigen::MatrixXd 				mDecoded;
// 	CyclizedMotion*					mDecodeData;

// 	std::vector<Eigen::Vector3d> 	mColors;

// 	p::object nn_module;
// 	bool mMotionLoaded;
// };
// #endif
