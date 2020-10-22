#include "MotionWidget.h"
#include "SkeletonBuilder.h"
#include "Character.h"
#include "Functions.h"
#include <GL/glu.h>
#include <iostream>
#include <Eigen/Geometry>
#include <QSlider>
#include <chrono>
#include <algorithm>
MotionWidget::
MotionWidget()
  :mCamera(new Camera(1000, 650)),mCurFrame(0),mPlay(false),
  mTrackCamera(false), mDrawBvh(true), mDrawSim(true), mDrawReg(true)
{
	this->startTimer(30);
}
MotionWidget::
MotionWidget(std::string motion, std::string ppo, std::string reg)
  :MotionWidget()
{
	mCurFrame = 0;
	mTotalFrame = 0;

	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

    mSkel_bvh = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    mSkel_reg = DPhy::SkeletonBuilder::BuildFromFile(path).first;
    mSkel_sim = DPhy::SkeletonBuilder::BuildFromFile(path).first;
	
	path = std::string(CAR_DIR)+std::string("/character/sandbag.xml");
	mSkel_obj = DPhy::SkeletonBuilder::BuildFromFile(path).first;
	
	if(ppo == "") {
		mRunSim = false;
		mDrawSim = false;
	} else {
		mRunSim = true;
	}
	if(reg == "") {
		mRunReg = false;
		mDrawReg = false;
	} else {
		mRunReg = true;
	}
	mPoints.setZero();
    path = std::string(CAR_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");
    DPhy::Character* ref = new DPhy::Character(path);
    mReferenceManager = new DPhy::ReferenceManager(ref);
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + motion);
    if(mRunReg) {
    	mRegressionMemory = new DPhy::RegressionMemory();
		mReferenceManager->SetRegressionMemory(mRegressionMemory);
    }

    if(mRunSim) {
	    path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(ppo, '/')[0] + std::string("/");
	    if(mRunReg)
	    	mReferenceManager->InitOptimization(1, path, true);
	    else
	    	mReferenceManager->InitOptimization(1, path);
	    mReferenceManager->LoadAdaptiveMotion("ex_0");
	    mDrawReg = true;

    } else if(mRunReg) {
    	mReferenceManager->InitOptimization(1, "", true);
    }

	v_param.resize(mReferenceManager->GetParamGoal().rows());
    v_param.setZero();

    std::vector<Eigen::VectorXd> pos;
    double phase = 0;
    for(int i = 0; i < 10000; i++) {
        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, false);
        p(3) -= 0.75; 
        pos.push_back(p);
        phase += mReferenceManager->GetTimeStep(phase, false);
    }

    UpdateMotion(pos, 0);
	initNetworkSetting(ppo, reg);


	DPhy::SetSkeletonColor(mSkel_bvh, Eigen::Vector4d(235./255., 73./255., 73./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_reg, Eigen::Vector4d(87./255., 235./255., 87./255., 1.0));
	DPhy::SetSkeletonColor(mSkel_sim, Eigen::Vector4d(235./255., 235./255., 235./255., 1.0));

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
MotionWidget::
initNetworkSetting(std::string ppo, std::string reg) {

    Py_Initialize();
    np::initialize();
    try {
    	if(reg != "") {
			p::object reg_main = p::import("regression");
	        this->mRegression = reg_main.attr("Regression")();
	        std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
	        this->mRegression.attr("initRun")(path, mReferenceManager->GetParamGoal().rows() + 1, mReferenceManager->GetDOF() + 1);
			mRegressionMemory->LoadParamSpace(path + "param_space");
	        mParamRange = mReferenceManager->GetParamRange();
	       
	        path = std::string(CAR_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
			mRegressionMemory->SaveContinuousParamSpace(path + "param_cspace");
    	}
    	if(ppo != "") {
    		this->mController = new DPhy::Controller(mReferenceManager, true, mRunReg, true);

    		p::object ppo_main = p::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			std::string path = std::string(CAR_DIR)+ std::string("/network/output/") + ppo;
			this->mPPO.attr("initRun")(path,
									   this->mController->GetNumState(), 
									   this->mController->GetNumAction());
			RunPPO();
			
    	}
    
    } catch (const p::error_already_set&) {
        PyErr_Print();
    }    
}
void 
MotionWidget::
setValue(const int &x){
	auto slider = qobject_cast<QSlider*>(sender());
    auto i = slider->property("i").toInt();
    v_param(i) =  x;
}
void 
MotionWidget::
UpdateRandomParam(const bool& pressed) {
	// int dof = mReferenceManager->GetDOF() + 1;

	// std::vector<Eigen::VectorXd> cps;
	// for(int k = 0; k < mReferenceManager->GetNumCPS() ; k++) {
	// 	cps.push_back(Eigen::VectorXd::Zero(dof));
	// }

	// std::vector<std::vector<double>> loss;
	// std::vector<std::vector<std::vector<double>>> loss_bottom;

	// for(int i = 0; i < 9; i++) {
	// 	std::vector<double> loss_row;
	// 	std::vector<std::vector<double>> loss_bottom_row;
	// 	for(int j = 0; j < 10; j++) {
	// 		loss_row.push_back(0);
	// 		std::vector<double> rec;
	// 		loss_bottom_row.push_back(rec);
	// 	}
	// 	loss.push_back(loss_row);
	// 	loss_bottom.push_back(loss_bottom_row);
	// } 
	// for(int j = 0; j < 100000; j++) {
	// 	Eigen::VectorXd tp_denorm = mRegressionMemory->UniformSample(0);
	// 	Eigen::VectorXd tp = mRegressionMemory->Normalize(tp_denorm);
	// 	auto n_params = mRegressionMemory->GetNearestParams(tp, 5);
	// 	double dist = 0;
	// 	for(int k = 0; k <= n_params.size(); k++) {
	// 		dist += n_params[k].first;
	// 	}
	// 	int y = (int) std::floor(dist);
	// 	if(y > 9)
	// 		y = 9;
	// 	int x = mRegressionMemory->GetNeighborParams(tp).size();
	// 	if(x > 8)
	// 		x = 8;
	// 	for(int k = 0; k < mReferenceManager->GetNumCPS(); k++) {
	// 	    Eigen::VectorXd input(mRegressionMemory->GetDim() + 1);
	// 	    input << k, tp;
	// 	    p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
		    
	// 	    np::ndarray na = np::from_object(a);
	// 	    cps[k] = DPhy::toEigenVector(na, dof);
	// 	}

	// 	mReferenceManager->LoadAdaptiveMotion(cps);

	// 	double phase = 0;
	// 	Eigen::VectorXd headRoot(6);
	// 	headRoot = mReferenceManager->GetPosition(phase, false).segment<6>(0);

	// 	for(int k = 0; k < 100; k++) {

	// 	        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        
	// 	        if(phase >= 18) {
	// 	      		mSkel_reg->setPositions(p);
	// 				mSkel_reg->computeForwardKinematics(true,false,false);

	// 	       		Eigen::Vector3d hand = mSkel_reg->getBodyNode("RightHand")->getWorldTransform().translation();
	// 				Eigen::Vector3d root_new = headRoot.segment<3>(0);
	// 				root_new = DPhy::projectToXZ(root_new);
	// 				Eigen::AngleAxisd aa(root_new.norm(), root_new.normalized());
	// 				Eigen::Vector3d dir = Eigen::Vector3d(tp_denorm(0), 0, - sqrt(1 - tp_denorm(0)*tp_denorm(0)));
	// 				dir.normalize();
	// 				dir *= tp_denorm(2);
	// 				Eigen::Vector3d goal_hand = aa * dir + headRoot.segment<3>(3);
	// 				goal_hand(1) = tp_denorm(1);
	// 				Eigen::Vector3d hand_diff = goal_hand - hand;
	// 				loss[x][y] += hand_diff.norm();
	// 				loss_bottom[x][y].push_back(hand_diff.norm());
	// 				break;
	// 			}
	// 	        phase += mReferenceManager->GetTimeStep(phase, true);
	// 	}
	// 	if(j % 100 == 0)
	// 		std::cout << j << std::endl;
	// }
	// std::string path =std::string(CAR_DIR)+ std::string("/utils/plot3d");
	// std::ofstream ofs(path);

	// for(int i = 0; i < 9; i++) {
	// 	for(int j = 0; j < 10; j++) {
	// 		if(loss_bottom[i][j].size() < 50)
	// 			continue;

	// 		std::vector<double> l = loss_bottom[i][j];
	// 		std::sort(l.begin(), l.end());
	// 		double loss_mean = 0;
	// 		int bottom_10 = loss_bottom[i][j].size() * 0.1;
	// 		for(int k = l.size() - 1; k > l.size() - bottom_10 - 1; k--) {
	// 			loss_mean += l[k];
	// 		}
	// 		ofs << i << " " << j << ": " << loss_bottom[i][j].size()  << ", " <<  
	// 					 loss[i][j] / loss_bottom[i][j].size() << " " << loss_mean / bottom_10 << std::endl;
	// 	}
	// }
	// ofs.close();
	if(mRunReg) {
	   	Eigen::VectorXd tp_denorm = mRegressionMemory->UniformSample(3);
	   	Eigen::VectorXd tp = mRegressionMemory->Normalize(tp_denorm);
	    int dof = mReferenceManager->GetDOF() + 1;
	    auto pairs = mRegressionMemory->GetNearestParams(tp, 10); 
	    for(int i = 0; i < pairs.size(); i++) {
	    	std::cout << pairs[i].first<< " " ;
	    }
	    std::cout << std::endl;
	    std::vector<Eigen::VectorXd> cps;
	    for(int i = 0; i < mReferenceManager->GetNumCPS() ; i++) {
	        cps.push_back(Eigen::VectorXd::Zero(dof));
	    }
	    for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
	        Eigen::VectorXd input(mRegressionMemory->GetDim() + 1);
	        input << j, tp;
	        p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
	    
	        np::ndarray na = np::from_object(a);
	        cps[j] = DPhy::toEigenVector(na, dof);
	    }

	    mReferenceManager->LoadAdaptiveMotion(cps);
	    
	    double phase = 0;
		Eigen::VectorXd headRoot(6);
		headRoot = mReferenceManager->GetPosition(phase, false).segment<6>(0);

		for(int k = 0; k < 100; k++) {

		    Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        
		    if(phase >= 18) {
		      	mSkel_reg->setPositions(p);
				mSkel_reg->computeForwardKinematics(true,false,false);

		       	Eigen::Vector3d hand = mSkel_reg->getBodyNode("RightHand")->getWorldTransform().translation();
				Eigen::Vector3d root_new = headRoot.segment<3>(0);
				root_new = DPhy::projectToXZ(root_new);
				Eigen::AngleAxisd aa(root_new.norm(), root_new.normalized());
				Eigen::Vector3d dir = Eigen::Vector3d(tp_denorm(0), 0, - sqrt(1 - tp_denorm(0)*tp_denorm(0)));
				dir.normalize();
				dir *= tp_denorm(2);
				Eigen::Vector3d goal_hand = aa * dir + headRoot.segment<3>(3);
				goal_hand(1) = tp_denorm(1);

				mPoints = goal_hand;
				mPoints(0) += 0.75;
				break;
			}
			phase += mReferenceManager->GetTimeStep(phase, true);

		}
	    if(!mRunSim) {

		    std::vector<Eigen::VectorXd> pos;
		    double phase = 0;

		    Eigen::VectorXd headRoot(6);
		    bool flag_test = false;
			headRoot = mReferenceManager->GetPosition(phase, false).segment<6>(0);

		    for(int i = 0; i < 500; i++) {

		        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        p(3) += 0.75;
		      	pos.push_back(p);
		        phase += mReferenceManager->GetTimeStep(phase, true);
		    }
		    mTotalFrame = 500;
		    Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
			root_bvh(3) += 0.75;
			pos = DPhy::Align(pos, root_bvh);

		    UpdateMotion(pos, 2);
	    } else {
	     	mTotalFrame = 0;
	     	mController->SetGoalParameters(tp_denorm);
			RunPPO();
	    }
	}
}
void 
MotionWidget::
UpdateParam(const bool& pressed) {
	std::cout << v_param.transpose() << std::endl;
	if(mRunReg) {
		Eigen::VectorXd tp = v_param * 0.1;
	    tp = mRegressionMemory->GetNearestParams(tp, 1)[0].second.param_normalized;
	    Eigen::VectorXd tp_denorm = mRegressionMemory->Denormalize(tp);
	    int dof = mReferenceManager->GetDOF() + 1;
	    auto pairs = mRegressionMemory->GetNearestParams(tp, 10); 
	    for(int i = 0; i < pairs.size(); i++) {
	    	std::cout << pairs[i].first<< " " ;
	    }
	    std::cout << std::endl;
	    std::vector<Eigen::VectorXd> cps;
	    for(int i = 0; i < mReferenceManager->GetNumCPS() ; i++) {
	        cps.push_back(Eigen::VectorXd::Zero(dof));
	    }
	    for(int j = 0; j < mReferenceManager->GetNumCPS(); j++) {
	        Eigen::VectorXd input(mRegressionMemory->GetDim() + 1);
	        input << j, tp;
	        p::object a = this->mRegression.attr("run")(DPhy::toNumPyArray(input));
	    
	        np::ndarray na = np::from_object(a);
	        cps[j] = DPhy::toEigenVector(na, dof);
	    }

	    mReferenceManager->LoadAdaptiveMotion(cps);
	    
	    double phase = 0;
		Eigen::VectorXd headRoot(6);
		headRoot = mReferenceManager->GetPosition(phase, false).segment<6>(0);

		for(int k = 0; k < 100; k++) {

		    Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        
		    if(phase >= 18) {
		      	mSkel_reg->setPositions(p);
				mSkel_reg->computeForwardKinematics(true,false,false);

		       	Eigen::Vector3d hand = mSkel_reg->getBodyNode("RightHand")->getWorldTransform().translation();
				Eigen::Vector3d root_new = headRoot.segment<3>(0);
				root_new = DPhy::projectToXZ(root_new);
				Eigen::AngleAxisd aa(root_new.norm(), root_new.normalized());
				Eigen::Vector3d dir = Eigen::Vector3d(tp_denorm(0), 0, - sqrt(1 - tp_denorm(0)*tp_denorm(0)));
				dir.normalize();
				dir *= tp_denorm(2);
				Eigen::Vector3d goal_hand = aa * dir + headRoot.segment<3>(3);
				goal_hand(1) = tp_denorm(1);

				mPoints = goal_hand;
				mPoints(0) += 0.75;
				break;
			}
			phase += mReferenceManager->GetTimeStep(phase, true);

		}
	    if(!mRunSim) {

		    std::vector<Eigen::VectorXd> pos;
		    double phase = 0;

		    Eigen::VectorXd headRoot(6);
		    bool flag_test = false;
			headRoot = mReferenceManager->GetPosition(phase, false).segment<6>(0);

		    for(int i = 0; i < 500; i++) {

		        Eigen::VectorXd p = mReferenceManager->GetPosition(phase, true);
		        p(3) += 0.75;
		      	pos.push_back(p);
		        phase += mReferenceManager->GetTimeStep(phase, true);
		    }
		    mTotalFrame = 500;
		    Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
			root_bvh(3) += 0.75;
			pos = DPhy::Align(pos, root_bvh);

		    UpdateMotion(pos, 2);
	    } else {
	     	mTotalFrame = 0;
	     	mController->SetGoalParameters(tp_denorm);
			RunPPO();
	    }
	}
}
void
MotionWidget::
RunPPO() {
	std::vector<Eigen::VectorXd> pos_reg;
	std::vector<Eigen::VectorXd> pos_sim;
	// std::vector<Eigen::VectorXd> pos_obj;

	int count = 0;
	mController->Reset(false);
	while(!this->mController->IsTerminalState()) {
		Eigen::VectorXd state = this->mController->GetState();

		p::object a = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
		np::ndarray na = np::from_object(a);
		Eigen::VectorXd action = DPhy::toEigenVector(na,this->mController->GetNumAction());

		this->mController->SetAction(action);
		this->mController->Step();
		count += 1;
	}

	for(int i = 0; i <= count; i++) {

		Eigen::VectorXd position = this->mController->GetPositions(i);
		Eigen::VectorXd position_reg = this->mController->GetTargetPositions(i);
		//Eigen::VectorXd position_obj = this->mController->GetObjPositions(i);

		position(3) += 0.75;
		position_reg(3) += 0.75;
		//position_obj(3) += 0.75;

		pos_reg.push_back(position_reg);
		pos_sim.push_back(position);
		//pos_obj.push_back(position_obj);
	}
	Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
	root_bvh(3) += 0.75;
	pos_sim =  DPhy::Align(pos_sim, root_bvh);
	pos_reg =  DPhy::Align(pos_reg, root_bvh);

	UpdateMotion(pos_sim, 1);
	UpdateMotion(pos_reg, 2);
	//UpdateMotion(pos_obj, 3);

}
void
MotionWidget::
initializeGL()
{
	glClearColor(1,1,1,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	mCamera->Apply();
}
void
MotionWidget::
resizeGL(int w,int h)
{
	glViewport(0, 0, w, h);
	mCamera->SetSize(w, h);
	mCamera->Apply();
}
void
MotionWidget::
SetFrame(int n)
{
	if(mDrawBvh && n < mMotion_bvh.size()) {
    	mSkel_bvh->setPositions(mMotion_bvh[n]);
	}
	if(mDrawSim && n < mMotion_sim.size()) {
    	mSkel_sim->setPositions(mMotion_sim[n]);
    //	mSkel_obj->setPositions(mMotion_obj[n]);

	}
	if(mDrawReg && n < mMotion_reg.size()) {
    	mSkel_reg->setPositions(mMotion_reg[n]);
    	// mPoints = mMotion_points[n];
	}
}
void
MotionWidget::
DrawSkeletons()
{

	if(mDrawBvh)
		GUI::DrawSkeleton(this->mSkel_bvh, 0);
	if(mDrawSim) {
		GUI::DrawSkeleton(this->mSkel_sim, 0);
	//	GUI::DrawSkeleton(this->mSkel_obj, 0);
	}
	if(mDrawReg) {
		GUI::DrawSkeleton(this->mSkel_reg, 0);
		// if(!mRunSim) {
			GUI::DrawPoint(mPoints, Eigen::Vector3d(1.0, 0.0, 0.0), 10);
	//	}
	}

}	
void
MotionWidget::
DrawGround()
{
	Eigen::Vector3d com_root;
	com_root = this->mSkel_bvh->getRootBodyNode()->getCOM();
	if(mRunReg) {
		com_root = 0.5 * com_root + 0.5 * this->mSkel_reg->getRootBodyNode()->getCOM();
	} else if(mRunSim) {
			com_root = 0.5 * com_root + 0.5 * this->mSkel_sim->getRootBodyNode()->getCOM();	
	}
	GUI::DrawGround((int)com_root[0], (int)com_root[2], 0);
}
void
MotionWidget::
paintGL()
{

	glClearColor(1.0, 1.0, 1.0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	initLights();
	glEnable(GL_LIGHTING);

	mCamera->Apply();
	// Eigen::Vector3d com_root = mSkel->getRootBodyNode()->getCOM();
	// Eigen::Vector3d com_front = mSkel->getRootBodyNode()->getTransform()*Eigen::Vector3d(0.0, 0.0, 2.0);

	// if(this->mTrackCamera){
	// 	Eigen::Vector3d com = mSkel->getRootBodyNode()->getCOM();
	// 	Eigen::Isometry3d transform = mSkel->getRootBodyNode()->getTransform();
	// 	com[1] = 0.8;

	// 	Eigen::Vector3d camera_pos;
	// 	camera_pos << -3, 1, 1.5;
	// 	camera_pos = camera_pos + com;
	// 	camera_pos[1] = 2;

	// 	mCamera->SetCenter(com);
	// }
	// initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	// glEnable(GL_LIGHTING);
	// mCamera->Apply();

	DrawGround();
	DrawSkeletons();

}
void
MotionWidget::
initLights()
{

	static float ambient[]           	 = {0.4, 0.4, 0.4, 1.0};
	static float diffuse[]             = {0.4, 0.4, 0.4, 1.0};
	static float front_mat_shininess[] = {60.0};
	static float front_mat_specular[]  = {0.2, 0.2,  0.2,  1.0};
	static float front_mat_diffuse[]   = {0.2, 0.2, 0.2, 1.0};
	static float lmodel_ambient[]      = {0.2, 0.2,  0.2,  1.0};
	static float lmodel_twoside[]      = {GL_TRUE};

	GLfloat position[] = {0.0, 1.0, 1.0, 0.0};
	GLfloat position1[] = {0.0, 1.0, -1.0, 0.0};

	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  lmodel_ambient);
	glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);

	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  front_mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   front_mat_diffuse);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);

	glEnable(GL_FOG);
	GLfloat fogColor[] = {200.0/256.0,200.0/256.0,200.0/256.0,1};
	glFogfv(GL_FOG_COLOR,fogColor);
	glFogi(GL_FOG_MODE,GL_LINEAR);
	glFogf(GL_FOG_DENSITY,0.05);
	glFogf(GL_FOG_START,20.0);
	glFogf(GL_FOG_END,40.0);
}
void
MotionWidget::
timerEvent(QTimerEvent* event)
{
	if(mPlay && mCurFrame < mTotalFrame) {
		mCurFrame += 1;
	} 
	SetFrame(this->mCurFrame);
	update();

}
void
MotionWidget::
toggleDrawBvh() {
	mDrawBvh = !mDrawBvh;

}
void
MotionWidget::
toggleDrawReg() {
	mDrawReg = !mDrawReg;

}
void
MotionWidget::
toggleDrawSim() {
	if(mRunSim)
		mDrawSim = !mDrawSim;

}
// void
// MotionWidget::
// toggleDraw(int type) {
// 	if(type == 0) {
// 		mDrawBvh != mDrawBvh;
// 	}
// 	if(type == 1) {
// 		mDrawSim != mDrawSim;
// 	}
// 	if(type == 2) {
// 		mDrawReg != mDrawReg;
// 	}
// }
void
MotionWidget::
keyPressEvent(QKeyEvent *event)
{
	if(event->key() == Qt::Key_Escape){
		exit(0);
	}
	if(event->key() == Qt::Key_Space){
		mPlay = !mPlay;
		if(mPlay)
			std::cout << "Play." << std::endl;
		else 
			std::cout << "Pause." << std::endl;
	}
}
void
MotionWidget::
mousePressEvent(QMouseEvent* event)
{
	mIsDrag = true;
	mButton = event->button();
	mPrevX = event->x();
	mPrevY = event->y();
}
void
MotionWidget::
mouseMoveEvent(QMouseEvent* event)
{
	if(!mIsDrag)
	return;

	if (mButton == Qt::MidButton)
		mCamera->Translate(event->x(),event->y(),mPrevX,mPrevY);
	else if(mButton == Qt::LeftButton)
		mCamera->Rotate(event->x(),event->y(),mPrevX,mPrevY);

	mPrevX = event->x();
	mPrevY = event->y();
	update();
}
void
MotionWidget::
mouseReleaseEvent(QMouseEvent* event)
{
	mIsDrag = false;
	mButton = Qt::NoButton;
	update();
}
void
MotionWidget::
wheelEvent(QWheelEvent *event)
{
	if(event->angleDelta().y()>0)
	mCamera->Pan(0,-5,0,0);
	else
	mCamera->Pan(0,5,0,0);
	update();
}
void 
MotionWidget::
UpdateMotion(std::vector<Eigen::VectorXd> motion, int type)
{
	if(type == 0) {
		mMotion_bvh = motion;	
	}
	else if(type == 1) {
		mMotion_sim = motion;		
	}
	else if(type == 2) {
		mMotion_reg = motion;	
	}
	else if(type == 3) {
		mMotion_obj = motion;	
	} else if(type == 4) {
		mMotion_points = motion;
	}
	mCurFrame = 0;
	if(mTotalFrame == 0)
		mTotalFrame = motion.size();
	else if(mTotalFrame > motion.size()) {
		mTotalFrame = motion.size();
	}
}
void
MotionWidget::
NextFrame()
{ 
	if(!mPlay) {
		this->mCurFrame += 1;
		this->SetFrame(this->mCurFrame);
	}
}
void
MotionWidget::
PrevFrame()
{
	if(!mPlay && mCurFrame > 0) {
		this->mCurFrame -= 1;
		this->SetFrame(this->mCurFrame);
	}
}
void
MotionWidget::
Reset()
{
	this->mCurFrame = 0;
	this->SetFrame(this->mCurFrame);
}
void 
MotionWidget::
togglePlay() {
	mPlay = !mPlay;
}
