#include "MainWindow.h"
#include "SkeletonBuilder.h"
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
MainWindow(std::string motion, std::string network)
{   
    MainWindow();
    initLayoutSetting();
    initNetworkSetting(motion, network);

}
bool cmp(const Eigen::VectorXd &p1, const Eigen::VectorXd &p2){
    for(int i = 0; i < p1.rows(); i++) {
        if(p1(i) < p2(i))
            return true;
        else if(p1(i) > p2(i))
            return false;
    }
    return false;
}
void
MainWindow::
initLayoutSetting() {
    mMainLayout = new QHBoxLayout();
    this->setMaximumSize(1300,750);
    this->setMinimumSize(1300,750);

    std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

    auto skel_bvh = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    auto skel_reg = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    auto skel_sim = DPhy::SkeletonBuilder::BuildFromFile(path).first;

    QVBoxLayout *motionlayout = new QVBoxLayout();

    mMotionWidget = new MotionWidget(skel_bvh, skel_reg, skel_sim);
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
            connect (param, SIGNAL(valueChanged(int)), this, SLOT(setValueX(const int&)));
        else
            connect (param, SIGNAL(valueChanged(int)), this, SLOT(setValueY(const int&)));

    }

    mParamlayout->addStretch(1);

    mParamlayout->addLayout(mParamFormlayout);
    mParamlayout->addStretch(1);

    mButton = new QPushButton("set", this);
    mButton->setStyleSheet("margin-bottom: 10px;"
                           "padding: 5px;");
    mParamlayout->addWidget(mButton);
    connect (mButton, SIGNAL(clicked(bool)), this, SLOT(UpdateParam(const bool&)));

    this->setCentralWidget(new QWidget());
    this->centralWidget()->setLayout(mMainLayout);
    mMainLayout->addLayout(motionlayout);
   
    mMainLayout->addStretch(1);
    mMainLayout->addLayout(mParamlayout);
    mMainLayout->addStretch(1);

}
void
MainWindow::
initNetworkSetting(std::string motion, std::string network) {
    v_param.resize(2);
    v_param.setZero();

    std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
    DPhy::Character* ref = new DPhy::Character(path);
    mReferenceManager = new DPhy::ReferenceManager(ref);
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);
    
    path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(network, '/')[0] + std::string("/");
    mReferenceManager->InitOptimization(1, path);
    mReferenceManager->LoadAdaptiveMotion("");

    std::vector<Eigen::VectorXd> pos;
    double phase = 0;
    for(int i = 0; i < 500; i++) {
        Eigen::VectorXd p =mReferenceManager->GetPosition(phase, false);
        p(3) -= 0.75; 
        pos.push_back(p);
        phase += mReferenceManager->GetTimeStep(phase, false);
        if(phase > mReferenceManager->GetPhaseLength())
            break;
    }
    mMotionWidget->UpdateMotion(pos, 0);

    Py_Initialize();
    np::initialize();
    try {
        p::object reg_main = p::import("regression");
        this->mRegression = reg_main.attr("Regression")();
        path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(network, '/')[0] + std::string("/");
        this->mRegression.attr("initRun")(path, mReferenceManager->GetTargetBase().rows() + 1, ref->GetSkeleton()->getNumDofs() + 1);

        path = path + "boundary";
        char buffer[256];

        std::ifstream is;
        is.open(path);

        int param_dof = v_param.rows();
        while(!is.eof()) {
            Eigen::VectorXd tp(param_dof);
            for(int j = 0; j < param_dof; j++) {
                is >> buffer;
                tp[j] = atof(buffer);
            }
            //comma
            is >> buffer;

            Eigen::VectorXd tp2(param_dof);
            for(int j = 0; j < param_dof; j++) {
                is >> buffer;
                tp2[j] = atof(buffer);
            }
            if((tp - tp2).norm() < 1e-2) 
                break;
            Eigen::VectorXd tp_mean = (tp + tp2) * 0.5;

            mParamRange.push_back(tp_mean);

        }
        is.close();
        std::stable_sort(mParamRange.begin(), mParamRange.end(), cmp);

    
    } catch (const p::error_already_set&) {
        PyErr_Print();
    }    
}
void 
MainWindow::
setValueX(const int &x){
    v_param(0) = x;
}
void 
MainWindow::
setValueY(const int &y){
    v_param(1) = y;
}
void 
MainWindow::
UpdateParam(const bool& pressed) {

    Eigen::VectorXd tp(v_param.rows());
    int startIdx = 0, endIdx = mParamRange.size() - 1;
    for(int i = 0 ; i < tp.rows(); i++) {
        double min = mParamRange[startIdx](i);
        double max = mParamRange[endIdx](i);
        tp(i) = (max - min) * 0.1 * v_param(i) + min;

        for(int j = startIdx; j <= endIdx + 1; j++) {
            if(j == endIdx || mParamRange[j][i] > tp(i)) {
                endIdx = j-1;
                for(int k = endIdx; k >= startIdx; k--) {
                    if(mParamRange[endIdx][i] != mParamRange[k][i]) {
                        startIdx = k+1;
                        break;
                    }
                }
                break;
            }
        }
    }
    std::cout << tp.transpose() << std::endl;

    int dof = mReferenceManager->GetDOF() + 1;

    std::vector<Eigen::VectorXd> cps;
    for(int i = 0; i < mReferenceManager->GetNumCPS() ; i++) {
        cps.push_back(Eigen::VectorXd::Zero(dof));
    }
    for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
        Eigen::VectorXd input(mReferenceManager->GetTargetBase().rows() + 1);
        input << j, tp;
        p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
    
        np::ndarray na = np::from_object(a);
        cps[j] = DPhy::toEigenVector(na, dof);
    }
    mReferenceManager->LoadAdaptiveMotion(cps);
    std::vector<Eigen::VectorXd> pos;
    double phase = 0;
    for(int i = 0; i < 500; i++) {
        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
        p(3) += 0.75;
        pos.push_back(p);
        phase += mReferenceManager->GetTimeStep(phase, true);
        if(phase > mReferenceManager->GetPhaseLength())
            break;
    }
    mMotionWidget->UpdateMotion(pos, 2);
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