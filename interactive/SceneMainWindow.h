#ifndef __SCENE_MAIN_WINDOW_H__
#define __SCENE_MAIN_WINDOW_H__
#include <QMainWindow>
#include <QHBoxLayout>
#include <string>
#include <vector>
#include <QSlider>
#include <QPushButton>
#include "SceneMotionWidget.h"

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
class SceneMainWindow : public QMainWindow
{
    Q_OBJECT
    
signals:
	
public slots:
	void togglePlay(const bool& toggled);
public:
    SceneMainWindow();

protected:
	QHBoxLayout* mMainLayout;
	SceneMotionWidget* mMotionWidget;
	QPushButton* mButton;
	std::vector<QSlider*> mParams;

	void initLayoutSetting() ;

};
#endif