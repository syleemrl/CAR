#include "MainWindow.h"
#include <QtWidgets/QApplication>
#include <QFormLayout>
#include <iostream>
#include <QGLWidget>
#include <QLabel>
#include <QCheckBox>
MainWindow::
MainWindow() :QMainWindow()
{

    this->setWindowTitle("Motion Editing Renderer");
}
MainWindow::
MainWindow(std::vector<std::string> motion)
{   
    MainWindow();
    initLayoutSetting(motion);
}
MainWindow::
MainWindow(std::string motion, std::string ppo, std::string reg)
{   
    MainWindow();
    initLayoutSetting(motion, ppo, reg);
}
void
MainWindow::
initLayoutSetting(std::vector<std::string> motion) {
    mMainLayout = new QHBoxLayout();
    this->setMaximumSize(1300,750);
    this->setMinimumSize(1300,750);

    QVBoxLayout *motionlayout = new QVBoxLayout();

    mBVHWidget = new BVHWidget(motion);

    mBVHWidget->setMinimumSize(1000,650);
    mBVHWidget->setMaximumSize(1000,650);

    motionlayout->addWidget(mBVHWidget);

    QHBoxLayout *buttonlayout = new QHBoxLayout();
    buttonlayout->addStretch(1);

    QPushButton* button = new QPushButton("reset", this);
    connect(button, SIGNAL(clicked(bool)), mBVHWidget, SLOT(Reset())); 
    buttonlayout->addWidget(button);
    
    button = new QPushButton("prev", this);
    connect(button, SIGNAL(clicked(bool)), mBVHWidget, SLOT(PrevFrame())); 
    buttonlayout->addWidget(button); 

    button = new QPushButton("play", this);
    button->setCheckable(true);
    connect(button, SIGNAL(toggled(bool)), this, SLOT(togglePlay(const bool&))); 
    buttonlayout->addWidget(button); 

    button = new QPushButton("next", this);
    connect(button, SIGNAL(clicked(bool)), mBVHWidget, SLOT(NextFrame())); 
    buttonlayout->addWidget(button);    

    button = new QPushButton("save", this);
    connect(button, SIGNAL(clicked(bool)), this, SLOT(SaveScreenshot())); 
    buttonlayout->addWidget(button);    
    buttonlayout->addStretch(1);

    motionlayout->addLayout(buttonlayout);

   
    this->setCentralWidget(new QWidget());
    this->centralWidget()->setLayout(mMainLayout);
    mMainLayout->addLayout(motionlayout);
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

    QHBoxLayout *checkboxlayout = new QHBoxLayout();
    QCheckBox *checkbox = new QCheckBox("hide bvh", this);
    connect(checkbox, SIGNAL(clicked(bool)), mMotionWidget, SLOT(toggleDrawBvh())); 
    checkboxlayout->addWidget(checkbox);

    checkbox = new QCheckBox("hide sim", this);
    connect(checkbox, SIGNAL(clicked(bool)), mMotionWidget, SLOT(toggleDrawSim())); 
    checkboxlayout->addWidget(checkbox);
    
    checkbox = new QCheckBox("hide ref", this);
    connect(checkbox, SIGNAL(clicked(bool)), mMotionWidget, SLOT(toggleDrawReg())); 
    checkboxlayout->addWidget(checkbox);

    checkbox = new QCheckBox("hide exp", this);
    connect(checkbox, SIGNAL(clicked(bool)), mMotionWidget, SLOT(toggleDrawExp())); 
    checkboxlayout->addWidget(checkbox);

    checkboxlayout->addStretch(1);

    motionlayout->addLayout(checkboxlayout);
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

    button = new QPushButton("save", this);
    connect(button, SIGNAL(clicked(bool)), mMotionWidget, SLOT(Save())); 
    buttonlayout->addWidget(button);    
    buttonlayout->addStretch(1);

    motionlayout->addLayout(buttonlayout);

    QVBoxLayout *mParamlayout = new QVBoxLayout();
    std::vector<std::string> labels;
    labels.push_back("gravity");
    labels.push_back("mass");

    QFormLayout *mParamFormlayout = new QFormLayout();
    for(int i = 0; i < labels.size(); i++) {
        QSlider* param = new QSlider(Qt::Horizontal);
        param->setMinimum(0);
        param->setMaximum(20);
        param->setSingleStep(1);
        param->setProperty("i", i);
        mParams.push_back(param);
        
        mParamFormlayout->addRow(QString::fromStdString(labels[i]), param);
        connect (param, SIGNAL(valueChanged(int)), mMotionWidget, SLOT(setValue(const int&)));
    }

    mParamlayout->addStretch(1);

    mParamlayout->addLayout(mParamFormlayout);
    mParamlayout->addStretch(1);

    QHBoxLayout *mButtonlayout = new QHBoxLayout();

    QPushButton* rbutton = new QPushButton("random", this);
    rbutton->setStyleSheet("margin-bottom: 10px;"
                           "padding: 5px;");
    mButtonlayout->addWidget(rbutton);

    mButton = new QPushButton("set", this);
    mButton->setStyleSheet("margin-bottom: 10px;"
                           "padding: 5px;");
    mButtonlayout->addWidget(mButton);
    mParamlayout->addLayout(mButtonlayout);
    connect (rbutton, SIGNAL(clicked(bool)), mMotionWidget, SLOT(UpdateRandomParam(const bool&)));

    connect (mButton, SIGNAL(clicked(bool)), mMotionWidget, SLOT(UpdateParam(const bool&)));
//// next button
    QPushButton* pbutton = new QPushButton("prev", this);
    pbutton->setStyleSheet("margin-bottom: 10px;"
                           "padding: 5px;");
    mButtonlayout->addWidget(pbutton);
    connect (pbutton, SIGNAL(clicked(bool)), mMotionWidget, SLOT(UpdatePrevParam(const bool&)));

    QPushButton* nbutton = new QPushButton("next", this);
    nbutton->setStyleSheet("margin-bottom: 10px;"
                           "padding: 5px;");
    mButtonlayout->addWidget(nbutton);
    connect (nbutton, SIGNAL(clicked(bool)), mMotionWidget, SLOT(UpdateNextParam(const bool&)));
/////

    this->setCentralWidget(new QWidget());
    this->centralWidget()->setLayout(mMainLayout);
    mMainLayout->addLayout(motionlayout);
   
    mMainLayout->addStretch(1);
    mMainLayout->addLayout(mParamlayout);
    mMainLayout->addStretch(1);

}
void
MainWindow::
SaveScreenshot() 
{
    QWidget *w = QApplication::activeWindow();
    if(w) {
        static int count = 0;
        QPixmap p = QPixmap::grabWidget(w);
        p.save(QString("./screenshot%1.png").arg(count));
        count++;
    }
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

