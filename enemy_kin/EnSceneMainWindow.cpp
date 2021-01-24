#include "EnSceneMainWindow.h"
#include <QtWidgets/QApplication>
#include <QFormLayout>
#include <iostream>
#include <QGLWidget>
#include <QLabel>
#include <QCheckBox>
SceneMainWindow::
SceneMainWindow() :QMainWindow()
{
    this->setWindowTitle("Motion Editing Renderer");
    initLayoutSetting();
}
void
SceneMainWindow::
initLayoutSetting() {
    mMainLayout = new QHBoxLayout();
    this->setMaximumSize(1300,750);
    this->setMinimumSize(1300,750);

    QVBoxLayout *motionlayout = new QVBoxLayout();
    mMotionWidget = new SceneMotionWidget();

    mMotionWidget->setMinimumSize(1000,650);
    mMotionWidget->setMaximumSize(1000,650);

    QHBoxLayout *checkboxlayout = new QHBoxLayout();

    QCheckBox* checkbox = new QCheckBox("hide sim", this);
    connect(checkbox, SIGNAL(clicked(bool)), mMotionWidget, SLOT(toggleDrawSim())); 
    checkboxlayout->addWidget(checkbox);
    
    checkbox = new QCheckBox("hide ref", this);
    connect(checkbox, SIGNAL(clicked(bool)), mMotionWidget, SLOT(toggleDrawReg())); 
    checkboxlayout->addWidget(checkbox);

    checkboxlayout->addStretch(1);

    motionlayout->addLayout(checkboxlayout);
    motionlayout->addWidget(mMotionWidget);

    QHBoxLayout *buttonlayout = new QHBoxLayout();
    buttonlayout->addStretch(1);

    QPushButton* button = new QPushButton("RE", this);
    connect(button, SIGNAL(clicked(bool)), mMotionWidget, SLOT(RE())); 
    buttonlayout->addWidget(button);

    button = new QPushButton("reset", this);
    connect(button, SIGNAL(clicked(bool)), mMotionWidget, SLOT(Reset())); 
    buttonlayout->addWidget(button);
    
    button = new QPushButton("prev", this);
    connect(button, SIGNAL(clicked(bool)), mMotionWidget, SLOT(PrevFrame())); 
    buttonlayout->addWidget(button); 

    button = new QPushButton("play", this);
    button->setCheckable(true);
    connect(button, SIGNAL(toggled(bool)), this, SLOT(togglePlay(const bool&))); 
    buttonlayout->addWidget(button); 

    button = new QPushButton("next", this);
    connect(button, SIGNAL(clicked(bool)), mMotionWidget, SLOT(NextFrame())); 
    buttonlayout->addWidget(button);    

    // button = new QPushButton("save", this);
    // connect(button, SIGNAL(clicked(bool)), mMotionWidget, SLOT(Save())); 
    // buttonlayout->addWidget(button);    
    buttonlayout->addStretch(1);

    motionlayout->addLayout(buttonlayout);

    this->setCentralWidget(new QWidget());
    this->centralWidget()->setLayout(mMainLayout);
    mMainLayout->addLayout(motionlayout);
}
void 
SceneMainWindow::
togglePlay(const bool& toggled)
{
    auto button = qobject_cast<QPushButton*>(sender());
    if(toggled) {
        button->setText("pause");
    } else {
        button->setText("play");
    }
    mMotionWidget->togglePlay();

}

