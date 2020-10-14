#include "MainWindow.h"
#include <QtWidgets/QApplication>
#include <QFormLayout>
#include <iostream>
#include <QGLWidget>
#include <QLabel>
MainWindow::
MainWindow() :QMainWindow()
{

    this->setWindowTitle("Motion Editing Renderer");
}
MainWindow::
MainWindow(std::string motion, std::string ppo, std::string reg)
{   
    MainWindow();
    initLayoutSetting(motion, ppo, reg);
}
void
MainWindow::
initLayoutSetting(std::string motion, std::string ppo, std::string reg) {
    mMainLayout = new QHBoxLayout();
    this->setMaximumSize(1300,750);
    this->setMinimumSize(1300,750);

    QVBoxLayout *motionlayout = new QVBoxLayout();

    mMotionWidget = new MotionWidget(motion, ppo, reg);

    mMotionWidget->setMinimumSize(1000,650);
    mMotionWidget->setMaximumSize(1000,650);

    motionlayout->addWidget(mMotionWidget);
    // QSlider* frame = new QSlider(Qt::Horizontal);
    // frame->setMinimum(0);
    // frame->setMaximum(mReferenceManager->GetPhaseLength());
    // frame->setSingleStep(1);
    // motionlayout->addWidget(frame);

    QHBoxLayout *buttonlayout = new QHBoxLayout();
    buttonlayout->addStretch(1);

    QPushButton* button = new QPushButton("reset", this);
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
    buttonlayout->addStretch(1);

    motionlayout->addLayout(buttonlayout);

    QVBoxLayout *mParamlayout = new QVBoxLayout();
    std::vector<std::string> labels;
    labels.push_back("velocity");
    labels.push_back("height");
         
    QFormLayout *mParamFormlayout = new QFormLayout();
    for(int i = 0; i < labels.size(); i++) {
        QSlider* param = new QSlider(Qt::Horizontal);
        param->setMinimum(0);
        param->setMaximum(10);
        param->setSingleStep(1);

        mParams.push_back(param);
        
        mParamFormlayout->addRow(QString::fromStdString(labels[i]), param);
        if(i == 0)
            connect (param, SIGNAL(valueChanged(int)), mMotionWidget, SLOT(setValueX(const int&)));
        else
            connect (param, SIGNAL(valueChanged(int)), mMotionWidget, SLOT(setValueY(const int&)));

    }

    mParamlayout->addStretch(1);

    mParamlayout->addLayout(mParamFormlayout);
    mParamlayout->addStretch(1);

    mButton = new QPushButton("set", this);
    mButton->setStyleSheet("margin-bottom: 10px;"
                           "padding: 5px;");
    mParamlayout->addWidget(mButton);
    connect (mButton, SIGNAL(clicked(bool)), mMotionWidget, SLOT(UpdateParam(const bool&)));

    this->setCentralWidget(new QWidget());
    this->centralWidget()->setLayout(mMainLayout);
    mMainLayout->addLayout(motionlayout);
   
    mMainLayout->addStretch(1);
    mMainLayout->addLayout(mParamlayout);
    mMainLayout->addStretch(1);

}
void 
MainWindow::
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