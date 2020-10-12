#ifndef __MAIN_WINDOW_H__
#define __MAIN_WINDOW_H__
#include <QMainWindow>
#include <QHBoxLayout>
#include <string>
#include <vector>
#include <QSlider>
#include <QPushButton>
#include "MotionWidget.h"
#include "ReferenceManager.h"

#pragma push_macro("slots")
#undef slots
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#pragma pop_macro("slots")

namespace p = boost::python;
namespace np = boost::python::numpy;
struct Param
{
	Eigen::VectorXd paramMean;
};
class MainWindow : public QMainWindow
{
    Q_OBJECT
    
signals:
	
public slots:
	void setValueX(const int &x);
	void setValueY(const int &y);
	void UpdateParam(const bool& pressed);
public:
    MainWindow();
    MainWindow(std::string motion, std::string network);

protected:
	QHBoxLayout* mMainLayout;
	MotionWidget* mMotionWidget;
	QPushButton* mButton;
	std::vector<QSlider*> mParams;

	Eigen::VectorXd v_param;
	p::object mRegression;

	DPhy::ReferenceManager* mReferenceManager;
	dart::dynamics::SkeletonPtr mSkel;	
	std::vector<Eigen::VectorXd> mParamRange;
 	void initNetworkSetting(std::string motion, std::string network);
};
#endif