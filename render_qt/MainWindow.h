#ifndef __MAIN_WINDOW_H__
#define __MAIN_WINDOW_H__
#include <QMainWindow>
#include <QHBoxLayout>
#include <string>
#include <vector>
#include <QSlider>
#include <QPushButton>
#include "MotionWidget.h"
#include "BVHWidget.h"
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
	void togglePlay(const bool& toggled);
public:
    MainWindow();
    MainWindow(std::vector<std::string> motion);
    MainWindow(std::string motion, std::string ppo, std::string reg);

protected:
	QHBoxLayout* mMainLayout;
	MotionWidget* mMotionWidget;
	BVHWidget* mBVHWidget;
	QPushButton* mButton;
	std::vector<QSlider*> mParams;

	void initLayoutSetting(std::vector<std::string> motion);
	void initLayoutSetting(std::string motion, std::string ppo, std::string reg) ;

};
#endif