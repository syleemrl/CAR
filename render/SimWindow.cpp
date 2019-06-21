#include <GL/glew.h>
#include "SimWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include "SkeletonBuilder.h"
#include "Functions.h"
#include "Humanoid.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;


SimWindow::
SimWindow()
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false),mIsCapture(false)
	,mShowRef(true),mShowMod(true),mIsVelExist(false),mShowCharacter(true),mShowRootTraj(false)
{
	mWorld = std::make_shared<dart::simulation::World>();

	SkeletonPtr skel = DPhy::SkeletonBuilder::BuildFromFile("../../render/cart.xml");


	mWorld->addSkeleton(skel);
	mCurFrame = 0;
	mDisplayTimeout = 33;
}

SimWindow::
SimWindow(std::string filename)
	:GLUTWindow(),mTrackCamera(false),mIsRotate(false),mIsAuto(false),mIsCapture(false)
	,mShowRef(true),mShowMod(true),mIsVelExist(false),mShowCharacter(true),mShowRootTraj(false),mSkeletonDrawType(0)
{
	mWorld = std::make_shared<dart::simulation::World>();

	std::ifstream ifs(filename);
	if(!ifs.is_open()){
		std::cout << "File doesn't exist" << std::endl;
		exit(0);
	}

	// read a number of characters
	std::string line;
	std::getline(ifs, line);
	int nCharacters = atoi(line.c_str());

	// read skeleton file
	int nDof = 0;
	DPhy::Character* character;
	for(int i = 0; i < nCharacters; i++){
		std::string skelfilename;

		std::getline(ifs, skelfilename);
		SkeletonPtr skel = DPhy::SkeletonBuilder::BuildFromFile(skelfilename);
		nDof += skel->getNumDofs();
		mWorld->addSkeleton(skel);

		// for reference motion
		if(skel->getName() == "Humanoid"){
			SkeletonPtr refSkel = skel->clone("Ref");
			mWorld->addSkeleton(refSkel);
			// character = new DPhy::Humanoid(skelfilename);

			SkeletonPtr modSkel = skel->clone("Mod");
			mWorld->addSkeleton(modSkel);

            DPhy::SetSkeletonColor(skel, Eigen::Vector4d(0.83, 0.83, 0.88, 1.0));
            DPhy::SetSkeletonColor(refSkel, Eigen::Vector4d(92./255., 145./255., 245./255., 0.9));
            DPhy::SetSkeletonColor(modSkel, Eigen::Vector4d(235./255., 87./255., 87./255., 0.9));
//            DPhy::SetSkeletonColor(modSkel, Eigen::Vector4d(45./255., 95./255., 94./255., 0.9));
            //41,144,100
			//212,147,52
            //45,95,94

            //hs 하늘색 92,145,236

            //연보라 214,200,237
            //연초록(마이픽) 78,197,176
            //중간보라169,103,255
            //찐한초록(수환픽)0,105,34

//			DPhy::SetSkeletonColor(skel, Eigen::Vector4d(0.73, 0.73, 0.78, 1.0));
//			DPhy::SetSkeletonColor(refSkel, Eigen::Vector4d(21./255., 70./255., 103./255., 0.9));
//			DPhy::SetSkeletonColor(modSkel, Eigen::Vector4d(133./255., 193./255., 204./255., 0.9));
            //(11,60,93)
			//(133,193,204)

//            DPhy::SetSkeletonColor(skel, Eigen::Vector4d(0.83, 0.83, 0.88, 1.0));
//            DPhy::SetSkeletonColor(refSkel, Eigen::Vector4d(255./255., 238./255., 181./255., 0.9));
//            DPhy::SetSkeletonColor(modSkel, Eigen::Vector4d(188./255., 143./255., 143./255., 0.9));
			//Rosy Brown	188	143	143
			//Saddle Brown	139	69	19
//            DPhy::SetSkeletonColor(skel, Eigen::Vector4d(0.73, 0.73, 0.78, 1.0));
//            DPhy::SetSkeletonColor(refSkel, Eigen::Vector4d(235./255., 87./255., 87./255., 0.9));
//            DPhy::SetSkeletonColor(modSkel, Eigen::Vector4d(93./255., 176./255., 89./255., 0.9));
		}
	}
	// std::string motionfilename;
	// std::getline(ifs, motionfilename);
	// mBVH = new DPhy::BVH();
	// mBVH->Parse(std::string(DPHY_DIR) + std::string("/motion/merged_69.bvh"));
	// character->InitializeBVH(mBVH);
	// this->mCharacter= character;

	// read frame number
	std::getline(ifs, line);
	this->mTotalFrame = atoi(line.c_str());

	// read start time for ref motion
	std::getline(ifs, line);
	double starttime = atof(line.c_str());

	// read timestep
	std::getline(ifs, line);
	this->mTimeStep = atof(line.c_str());

	// read joint angles per frame
	for(int i = 0; i < this->mTotalFrame; i++){
		std::getline(ifs, line);
		Eigen::VectorXd record = DPhy::string_to_vectorXd(line, nDof);
		double t = i*this->mTimeStep;
		// Eigen::VectorXd refPos = character->GetTargetPositionsAndVelocitiesFromBVH(mBVH, starttime+t).first;
		// std::cout << i << " : " << refPos.segment<3>(3).transpose() << std::endl;

		// Eigen::VectorXd newRec(record.rows() + refPos.rows());
		// newRec << record, refPos;
		this->mRecords.push_back(record);
	}

	nDof = mWorld->getSkeleton("Humanoid")->getNumDofs();
    this->mIsRefExist = false;
	if(std::getline(ifs, line)){
		std::cout << line << std::endl;
		if( line == "Refs"){
			this->mIsRefExist = true;
			for(int i = 0; i < this->mTotalFrame; i++){
				std::getline(ifs, line);
				Eigen::VectorXd ref = DPhy::string_to_vectorXd(line, nDof);
				this->mRefRecords.push_back(ref);
			}
		}
	}

	this->mIsVelExist = false;
	if(line == "Vels"){
		this->mIsVelExist = true;
		for(int i = 0; i < this->mTotalFrame; i++){
			std::getline(ifs, line);
			Eigen::Vector3d vel = DPhy::string_to_vector3d(line);
			this->mVelRecords.push_back(vel);
		}
	}
	else if(std::getline(ifs, line)){
		std::cout << line << std::endl;
		if( line == "Vels"){
			this->mIsVelExist = true;
			for(int i = 0; i < this->mTotalFrame; i++){
				std::getline(ifs, line);
				Eigen::Vector3d vel = DPhy::string_to_vector3d(line);
				this->mVelRecords.push_back(vel);
			}
		}
	}

	this->mIsGoalExist = false;
	if(line == "Goals"){
		this->mIsGoalExist = true;
		for(int i = 0; i < this->mTotalFrame; i++){
			std::getline(ifs, line);
			Eigen::Vector3d goal = DPhy::string_to_vector3d(line);
			this->mGoalRecords.push_back(goal);
		}
	}
	else if(std::getline(ifs, line)){
		std::cout << line << std::endl;
		if( line == "Goals"){
			this->mIsGoalExist = true;
			for(int i = 0; i < this->mTotalFrame; i++){
				std::getline(ifs, line);
				Eigen::Vector3d goal = DPhy::string_to_vector3d(line);
				this->mGoalRecords.push_back(goal);
			}
		}
	}

	this->mIsModExist = false;
	if(line == "Mods"){
		this->mIsModExist = true;
		for(int i = 0; i < this->mTotalFrame; i++){
			std::getline(ifs, line);
			Eigen::VectorXd mod = DPhy::string_to_vectorXd(line, nDof);
			this->mModRecords.push_back(mod);
		}
	}
	else if(std::getline(ifs, line)){
		std::cout << line << std::endl;
		if( line == "Mods"){
			this->mIsModExist = true;
			for(int i = 0; i < this->mTotalFrame; i++){
				std::getline(ifs, line);
				Eigen::VectorXd mod = DPhy::string_to_vectorXd(line, nDof);
				this->mModRecords.push_back(mod);
			}
		}
	}

	this->mIsFootExist = false;
	if(line == "Foots"){
		this->mIsFootExist = true;
		for(int i = 0; i < this->mTotalFrame; i++){
			std::getline(ifs, line);
			Eigen::Vector2d foot = DPhy::string_to_vectorXd(line, 2);
			this->mFootRecords.push_back(foot);
		}
	}
	else if(std::getline(ifs, line)){
		std::cout << line << std::endl;
		if( line == "Foots"){
			this->mIsFootExist = true;
			for(int i = 0; i < this->mTotalFrame; i++){
				std::getline(ifs, line);
				Eigen::Vector2d foot = DPhy::string_to_vectorXd(line, 2);
				this->mFootRecords.push_back(foot);
			}
		}
	}

	if(line == "RefFoots"){
		for(int i = 0; i < this->mTotalFrame; i++){
			std::getline(ifs, line);
			Eigen::Vector2d foot = DPhy::string_to_vectorXd(line, 2);
			this->mRefFootRecords.push_back(foot);
		}
	}
	else if(std::getline(ifs, line)){
		std::cout << line << std::endl;
		if( line == "RefFoots"){
			for(int i = 0; i < this->mTotalFrame; i++){
				std::getline(ifs, line);
				Eigen::Vector2d foot = DPhy::string_to_vectorXd(line, 2);
				this->mRefFootRecords.push_back(foot);
			}
		}
	}


	mRootTrajectories.resize(this->mTotalFrame);
	mRootTrajectoriesRef.resize(this->mTotalFrame);
	for(int i = 0; i < this->mTotalFrame; i++){
		this->SetFrame(i);
		mRootTrajectories[i] = this->mWorld->getSkeleton("Humanoid")->getCOM();
		if(this->mIsRefExist)
			mRootTrajectoriesRef[i] = this->mWorld->getSkeleton("Ref")->getCOM();
	}

	mDisplayTimeout = 33;
	mCurFrame = 0;

	this->SetFrame(this->mCurFrame);

	ifs.close();

}

void
SimWindow::
SetFrame(int n)
{
	if( n < 0 || n >= this->mTotalFrame )
	{
		std::cout << "Frame exceeds limits" << std::endl;
		return;
	}

    int dof_index = 0;
	 for(int i = 0; i < this->mWorld->getNumSkeletons(); i++){
	 	SkeletonPtr skel = this->mWorld->getSkeleton(i);
	 	int dofs = skel->getNumDofs();
	 	Eigen::VectorXd pos = this->mRecords[n].segment(dof_index, dofs);
	 	skel->setPositions(pos);
	 	skel->computeForwardKinematics(true, false, false);
	 	// std::cout << skel->getCOM().transpose() << std::endl;
	 	dof_index += dofs;
	 }
	SkeletonPtr skel = this->mWorld->getSkeleton("Humanoid");
	Eigen::VectorXd pos;// = this->mRecords[n];
	// pos.setZero();
	// pos[4] = 1.0;
//	skel->setPositions(pos);
//	skel->computeForwardKinematics(true, false, false);

	// auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	// auto cg1 = collisionEngine->createCollisionGroup(mWorld->getSkeleton("Ground").get());
	// auto cg2 = collisionEngine->createCollisionGroup(mWorld->getSkeleton("Humanoid")->getBodyNode("FootR"));
	// dart::collision::CollisionOption option;
	// dart::collision::CollisionResult result;

	// bool collision = collisionEngine->collide(cg1.get(), cg2.get(), option, &result);
	// bool collision = mWorld->checkCollision(option, &result);
	// If the new object is not in collision

	// size_t collisionCount = result.getNumContacts();
	// std::cout << std::endl;
	// std::cout << collisionCount << std::endl;
	// for(auto bn : result.getCollidingBodyNodes()){
	// 	std::cout << bn->getSkeleton()->getName() << " - " << bn->getName() << std::endl;
	// }
	// for(size_t i = 0; i < collisionCount; ++i)
	// {
	//   const dart::collision::Contact& contact = result.getContact(i);
	//   std::cout << std::endl;
	//   std::cout << contact.collisionObject1->getShapeFrame()->getName() << std::endl;
	//   std::cout << contact.collisionObject2->getShapeFrame()->getName() << std::endl;
	// }

	if(this->mIsRefExist){
		skel = this->mWorld->getSkeleton("Ref");
		pos = this->mRefRecords[n];
		// Eigen::VectorXd refPos = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, n*0.03333+0.05).first;
		// pos.setZero();
		// pos[4] = 1.0;
		// pos[20] = 1.0;
		// pos[39] = 1.0;
		// pos[51] = 1.0;
		skel->setPositions(pos);
		skel->computeForwardKinematics(true, false, false);
	}

	if(this->mIsModExist){
		skel = this->mWorld->getSkeleton("Mod");
		pos = this->mModRecords[n];
		// Eigen::VectorXd refPos = mCharacter->GetTargetPositionsAndVelocitiesFromBVH(mBVH, n*0.03333+0.05).first;
		// pos.setZero();
		// pos[4] = 1.0;
		skel->setPositions(pos);
		skel->computeForwardKinematics(true, false, false);
	}

	// if(this->mIsFootExist){
		// skel = this->mWorld->getSkeleton("Humanoid");
		// // std::cout << this->mFootRecords[n].transpose() << std::endl;
		// if(this->mFootRecords[n][0] > 0)
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootL"), dart::Color::Black());
		// else
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootL"), Eigen::Vector3d(0.8, 0.8, 0.8));
		// if(this->mFootRecords[n][1] > 0)
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootR"), dart::Color::Black());
		// else
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootR"), Eigen::Vector3d(0.8, 0.8, 0.8));

		// skel = this->mWorld->getSkeleton("Ref");
		// if(this->mRefFootRecords[n][0] > 0)
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootL"), dart::Color::Black());
		// else
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootL"), dart::Color::Red());
		// if(this->mRefFootRecords[n][1] > 0)
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootR"), dart::Color::Black());
		// else
		// 	DPhy::SetBodyNodeColors(skel->getBodyNode("FootR"), dart::Color::Red());

	// }

	if(this->mTrackCamera){
		Eigen::Vector3d com = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
		Eigen::Isometry3d transform = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getTransform();
		com[1] = 0.8;

		Eigen::Vector3d camera_pos;
		// Eigen::Quaterniond t(Eigen::AngleAxisd(transform.linear()));
		// transform.linear() = DPhy::GetYRotation(t).toRotationMatrix();
		camera_pos << -3, 1, 1.5;
		// camera_pos = transform * camera_pos;
		camera_pos = camera_pos + com;
		camera_pos[1] = 2;

		// mCamera->SetCamera(com, camera_pos, Eigen::Vector3d::UnitY());
		mCamera->SetCenter(com);
	}



}
void
SimWindow::
NextFrame()
{ 
	this->mCurFrame+=1;
	this->mCurFrame %= this->mTotalFrame;
	// std::cout<<this->mCurFrame<<std::endl;
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
NextFrameRealTime()
{
	// int count = this->mDisplayTimeout/(this->mTimeStep*1000.);
	int count = 1;
	this->mCurFrame += count;
	this->mCurFrame %= this->mTotalFrame;
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
PrevFrame()
{
	this->mCurFrame-=1;
	if( this->mCurFrame < 0 ) this->mCurFrame = this->mTotalFrame -1;
	this->SetFrame(this->mCurFrame);
}
void
SimWindow::
DrawSkeletons()
{
	auto skel = this->mWorld->getSkeleton("Ref");
	if(mShowRef){
		GUI::DrawSkeleton(skel, this->mSkeletonDrawType);
		// foot height

		// const dart::dynamics::BodyNode *bn1, *bn2;
		// bn1 = skel->getBodyNode("FootL");
		// bn2 = skel->getBodyNode("FootEndL");

		// std::vector<Eigen::Vector3d> offsetList;
		// offsetList.resize(4);

		// offsetList[0] = Eigen::Vector3d(0.0375, -0.065, 0.025);
		// offsetList[1] = Eigen::Vector3d(-0.0375, -0.065, 0.025);
		// // offsetList[2] = Eigen::Vector3d(0.0375, 0.065, 0.025);
		// // offsetList[3] = Eigen::Vector3d(-0.0375, 0.065, 0.025);

		// // offsetList[4] = Eigen::Vector3d(0.0375, -0.025, 0.025);
		// // offsetList[5] = Eigen::Vector3d(-0.0375, -0.025, 0.025);
		// offsetList[2] = Eigen::Vector3d(0.0375, 0.025, 0.025);
		// offsetList[3] = Eigen::Vector3d(-0.0375, 0.025, 0.025);

		// std::vector<double> heightList;
		// heightList.resize(8);
		// for(int i = 0 ; i < 4; i++){
		// 	heightList[i] = (bn1->getWorldTransform()*offsetList[i])[1];
		// }
		// for(int i = 4 ; i < 8; i++){
		// 	heightList[i] = (bn2->getWorldTransform()*offsetList[i])[1];
		// }

		// double min = heightList[0];
		// int min_idx = 0;
		// for(int i = 1; i < 8; i++){
		// 	if(heightList[i] < min){
		// 		min = heightList[i];
		// 		min_idx = i;
		// 	}
		// }
		// // std::cout << min_idx << std::endl;
		// for(int i = 0; i < offsetList.size(); i++){
		// 	Eigen::Vector3d p;
		// 	if(i < 2)
		// 	{
		// 		p = bn1->getWorldTransform()*offsetList[i];
		// 	}
		// 	else
		// 	{
		// 		p = bn2->getWorldTransform()*offsetList[i];
		// 	}
		// 	glPushMatrix();
		// 	glTranslatef(p[0], p[1], p[2]);
		// 	glutSolidSphere(0.01, 20, 20);
		// 	glPopMatrix();		
		// }
	}
	skel = this->mWorld->getSkeleton("Mod");
	if(mShowMod){
		GUI::DrawSkeleton(skel, this->mSkeletonDrawType);
	}
	skel = this->mWorld->getSkeleton("Humanoid");
	if(mShowCharacter){
		GUI::DrawSkeleton(skel, this->mSkeletonDrawType);
		// if(this->mIsVelExist){
		// 	Eigen::Vector3d ori = skel->getRootBodyNode()->getCOM();
		// 	Eigen::Vector3d vel = this->mVelRecords[this->mCurFrame];
		// 	// Eigen::Quaterniond q(Eigen::AngleAxisd(goal, Eigen::Vector3d::UnitY()));
		// 	// Eigen::Vector3d goal_delta = q._transformVector(Eigen::Vector3d::UnitZ());
		// 	Eigen::Vector3d end = ori + vel;

		// 	glLineWidth(10.0);
		// 	glColor3f(0.2, 0.2, 0.2);
		// 	glBegin(GL_LINES);
		// 	glVertex3f(ori[0], ori[1], ori[2]);
		// 	glVertex3f(end[0], end[1], end[2]);
		// 	glEnd();
		// }



		// Eigen::Vector3d p0(0.0375, -0.025, 0.025);
		// Eigen::Vector3d p1(-0.0375, -0.025, 0.025);
		// Eigen::Vector3d p2(0.0375, 0.025, 0.025);
		// Eigen::Vector3d p3(-0.0375, 0.025, 0.025);

		// Eigen::Vector3d p0_l = bn1->getWorldTransform()*p0;
		// Eigen::Vector3d p1_l = bn1->getWorldTransform()*p1;
		// Eigen::Vector3d p2_l = bn1->getWorldTransform()*p2;
		// Eigen::Vector3d p3_l = bn1->getWorldTransform()*p3;

		// Eigen::Vector3d p0_r = bn2->getWorldTransform()*p0;
		// Eigen::Vector3d p1_r = bn2->getWorldTransform()*p1;
		// Eigen::Vector3d p2_r = bn2->getWorldTransform()*p2;
		// Eigen::Vector3d p3_r = bn2->getWorldTransform()*p3;

		// glPushMatrix();
		// glTranslatef(p0_l[0], p0_l[1], p0_l[2]);
		// glutSolidSphere(0.01, 20, 20);
		// glPopMatrix();
		// glPushMatrix();
		// glTranslatef(p1_l[0], p1_l[1], p1_l[2]);
		// glutSolidSphere(0.01, 20, 20);
		// glPopMatrix();
	}

    skel = this->mWorld->getSkeleton("BasketBall");
	// std::cout<<skel->getPositions().transpose(); //segment<3>(3).transpose();
	// std::cout<< " /// "<<this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM().transpose()<<std::endl;
    if(skel != nullptr)
		GUI::DrawSkeleton(skel);


}
void
SimWindow::
DrawGround()
{
	Eigen::Vector3d com_root = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
	double ground_height = this->mWorld->getSkeleton("Ground")->getRootBodyNode()->getCOM()[1]+0.5;
	GUI::DrawGround((int)com_root[0], (int)com_root[2], ground_height);
}
void
SimWindow::
Display() 
{

	glClearColor(0.9, 0.9, 0.9, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	Eigen::Vector3d com_root = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
	Eigen::Vector3d com_front = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getTransform()*Eigen::Vector3d(0.0, 0.0, 2.0);
	mCamera->Apply();
	// initLights(com_root[0], com_root[2]);
	
	glUseProgram(program);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPushMatrix();
    glScalef(1.0, -1.0, 1.0);
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	DrawSkeletons();
	glPopMatrix();
	initLights(com_root[0], com_root[2], com_front[0], com_front[2]);
	// glColor4f(0.7, 0.0, 0.0, 0.40);  /* 40% dark red floor color */
	DrawGround();
	DrawSkeletons();
	glDisable(GL_BLEND);


	if(mShowRootTraj){
		{
			// glDisable(GL_LIGHTING);
			glLineWidth(5.0);
			if(this->mIsRefExist){
				glColor3f(1.0,0.0,0.0);
				glBegin(GL_LINE_STRIP);
				for(int i = 0; i < this->mTotalFrame; i++){
    				glNormal3f(0.0, 1.0, 0.0);
					glVertex3f(this->mRootTrajectoriesRef[i][0], this->mRootTrajectoriesRef[i][1], this->mRootTrajectoriesRef[i][2]);
				}
				glEnd();
			}

			glColor3f(0.0,1.0,0.0);
			glBegin(GL_LINE_STRIP);
			for(int i = 0; i < this->mTotalFrame; i++){
    			glNormal3f(0.0, 1.0, 0.0);
				glVertex3f(this->mRootTrajectories[i][0], this->mRootTrajectories[i][1], this->mRootTrajectories[i][2]);
			}
			glEnd();
		}
	}
	{
		// glDisable(GL_LIGHTING);
		glLineWidth(10.0);
		if(this->mIsGoalExist){
            Eigen::Vector3d orange= dart::Color::Orange();
            glColor3f(orange[0],orange[1],orange[2]);
            glBegin(GL_TRIANGLES);
            glVertex3f(this->mGoalRecords[this->mCurFrame][0], 1.6, this->mGoalRecords[this->mCurFrame][2]);
            glVertex3f(this->mGoalRecords[this->mCurFrame][0], 1.97, this->mGoalRecords[this->mCurFrame][2]);
            glVertex3f(this->mGoalRecords[this->mCurFrame][0]+0.3, 1.85, this->mGoalRecords[this->mCurFrame][2]+0.3);
            glEnd();

            glPushMatrix();
            glTranslatef(this->mGoalRecords[this->mCurFrame][0], 1.0, this->mGoalRecords[this->mCurFrame][2]);
            glRotatef(90,1,0,0);
//            std::cout<<orange.transpose()<<std::endl; 1, 0.63, 0
            glColor3f(0.9,0.53,0.1);
            GUI::DrawCylinder(0.02, 2);
            glPopMatrix();

//			glColor3f(0.0, 0.0, 1.0);
//			glBegin(GL_LINE_STRIP);
//    		glNormal3f(0.0, 1.0, 0.0);
//			glVertex3f(this->mGoalRecords[this->mCurFrame][0], 0.0, this->mGoalRecords[this->mCurFrame][2]);
//			glVertex3f(this->mGoalRecords[this->mCurFrame][0], 2.0, this->mGoalRecords[this->mCurFrame][2]);
//			glEnd();
		}
	}


	// if(exist_humanoid)
	// {
	// 	glBegin(GL_LINES);
	// 	glLineWidth(3.0);
	// 	for(double z =-100.0;z<=100.0;z+=1.0){
	// 		glVertex3f(z,100,0);
	// 		glVertex3f(z,-100,0);
	// 	}
	// 	for(double y =-0.0;y<=3.0;y+=1.0){
	// 		glVertex3f(100,y,0);
	// 		glVertex3f(-100,y,0);
	// 	}
	// 	glEnd();
	// }


	glUseProgram(0);
	glutSwapBuffers();
	if(mIsCapture)
		Screenshot();
	// glutPostRedisplay();
}
void
SimWindow::
Keyboard(unsigned char key,int x,int y) 
{
	switch(key)
	{
		case '`' :mIsRotate= !mIsRotate;break;
		case '1' :mShowCharacter= !mShowCharacter;break;
		case '2' :mShowRef= !mShowRef;break;
		case '3' :mShowMod= !mShowMod;break;
		case '0' :mShowRootTraj= !mShowRootTraj;break;
		case '[': this->PrevFrame();break;
		case ']': this->NextFrame();break;
		case 'o': this->mCurFrame-=99; this->PrevFrame();break;
		case 'p': this->mCurFrame+=99; this->NextFrame();break;
		case 's': std::cout << this->mCurFrame << std::endl;break;
		case 'r': this->mCurFrame=0;this->SetFrame(this->mCurFrame);break;
		case 'C': mIsCapture = true; break;
		case 't': mTrackCamera = !mTrackCamera; this->SetFrame(this->mCurFrame); break;
		case 'T': this->mSkeletonDrawType++; this->mSkeletonDrawType %= 2; break;
		case ' ':
			mIsAuto = !mIsAuto;
			break;
		case 27: exit(0);break;
		default : break;
	}
	// this->SetFrame(this->mCurFrame);

	// glutPostRedisplay();
}
void
SimWindow::
Mouse(int button, int state, int x, int y) 
{
	if(button == 3 || button == 4){
		if (button == 3)
		{
			mCamera->Pan(0,-5,0,0);
		}
		else
		{
			mCamera->Pan(0,5,0,0);
		}
	}
	else{
		if (state == GLUT_DOWN)
		{
			mIsDrag = true;
			mMouseType = button;
			mPrevX = x;
			mPrevY = y;
		}
		else
		{
			mIsDrag = false;
			mMouseType = 0;
		}
	}

	// glutPostRedisplay();
}
void
SimWindow::
Motion(int x, int y) 
{
	if (!mIsDrag)
		return;

	int mod = glutGetModifiers();
	if (mMouseType == GLUT_LEFT_BUTTON)
	{
		// if(!mIsRotate)
		mCamera->Translate(x,y,mPrevX,mPrevY);
		// else
		// 	mCamera->Rotate(x,y,mPrevX,mPrevY);
	}
	else if (mMouseType == GLUT_RIGHT_BUTTON)
	{
		mCamera->Rotate(x,y,mPrevX,mPrevY);
		// switch (mod)
		// {
		// case GLUT_ACTIVE_SHIFT:
		// 	mCamera->Zoom(x,y,mPrevX,mPrevY); break;
		// default:
		// 	mCamera->Pan(x,y,mPrevX,mPrevY); break;		
		// }

	}
	mPrevX = x;
	mPrevY = y;
	// glutPostRedisplay();
}
void
SimWindow::
Reshape(int w, int h) 
{
	glViewport(0, 0, w, h);
	mCamera->Apply();
}
void
SimWindow::
Timer(int value) 
{
	if( mIsAuto )
		this->NextFrameRealTime();
	
	glutTimerFunc(mDisplayTimeout, TimerEvent,1);
	glutPostRedisplay();
}


void SimWindow::
Screenshot() {
  static int count = 0;
  const char directory[8] = "frames";
  const char fileBase[8] = "Capture";
  char fileName[32];

  boost::filesystem::create_directories(directory);
  std::snprintf(fileName, sizeof(fileName), "%s%s%s%.4d.png",
                directory, "/", fileBase, count++);
  int tw = glutGet(GLUT_WINDOW_WIDTH);
  int th = glutGet(GLUT_WINDOW_HEIGHT);

  glReadPixels(0, 0,  tw, th, GL_RGBA, GL_UNSIGNED_BYTE, &mScreenshotTemp[0]);

  // reverse temp2 temp1
  for (int row = 0; row < th; row++) {
    memcpy(&mScreenshotTemp2[row * tw * 4],
           &mScreenshotTemp[(th - row - 1) * tw * 4], tw * 4);
  }

  unsigned result = lodepng::encode(fileName, mScreenshotTemp2, tw, th);

  // if there's an error, display it
  if (result) {
    std::cout << "lodepng error " << result << ": "
              << lodepng_error_text(result) << std::endl;
    return ;
  } else {
    std::cout << "wrote screenshot " << fileName << "\n";
    return ;
  }
}
