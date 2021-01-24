#include "EnemyKinController.h"
#include <tinyxml.h>

namespace DPhy
{

EnemyKinController::EnemyKinController()
{
	std::string path = std::string(CAR_DIR)+std::string("/character/") + std::string(CHARACTER_TYPE) + std::string(".xml");
	this->mCharacter = new DPhy::Character(path);
	this->mCharacter_main_tmp = new DPhy::Character(path);

	mMotionFrames["box_move_back"] = 26;
	mMotionFrames["box_move_front"] = 29;
	mMotionFrames["box_move_left"] = 25;
	mMotionFrames["box_move_right"] = 26;
	mMotionFrames["pivot_mxm"] = 39;
	mMotionFrames["right_pivot_mxm"] = 37;
	mMotionFrames["box_idle"] = 67;

	mCurrentMotion = "box_idle";
	mNextMotion = "box_idle";
	mCurrentFrameOnPhase= 0;

    mReferenceManager = new DPhy::ReferenceManager(mCharacter);
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/")+ mCurrentMotion + std::string(".bvh"));

    mAlign = Eigen::Isometry3d::Identity();
    mTotalFrame = 0;
}

void EnemyKinController::Step(){
	//

}

void EnemyKinController::Reset(){
	//
}

void EnemyKinController::calculateAlign(){
	Eigen::VectorXd position = mCharacter->GetSkeleton()->getPositions();
	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(position.head<6>());

	Eigen::Isometry3d prev_cycle_end= T_current;
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/")+ mCurrentMotion + std::string(".bvh"));
    Eigen::VectorXd cycle_start_pos = mReferenceManager->GetPosition(mCurrentFrameOnPhase, false);
    Eigen::Isometry3d cycle_start = dart::dynamics::FreeJoint::convertToTransform(cycle_start_pos.head<6>());

    mAlign = prev_cycle_end* cycle_start.inverse();
	// Eigen::Isometry3d align= Eigen::Isometry3d::Identity();
	// align.linear() = Eigen::Matrix3d::Identity();
	// align.translation() = prev_cycle_end.translation()-cycle_start.translation();
	// align.translation()[1] = 0;
	// std::cout<<"same / prev end "<<prev_cycle_end.translation().transpose()<<" / cycle start: "<<cycle_start.translation().transpose()<<" / align: "<<align.translation().transpose()<<std::endl;

	// return align;
	// mAlign = align;
}

void EnemyKinController::Step(Eigen::VectorXd main_p){

	mCharacter_main_tmp->GetSkeleton()->setPositions(main_p);
	mCharacter_main_tmp->GetSkeleton()->computeForwardKinematics(true, true, false);

	Eigen::Vector3d main_root	= mCharacter_main_tmp->GetSkeleton()->getRootBodyNode()->getWorldTransform().translation();
	Eigen::Vector3d main_ls 	= mCharacter_main_tmp->GetSkeleton()->getBodyNode("LeftShoulder")->getWorldTransform().translation();
	Eigen::Vector3d main_rs 	= mCharacter_main_tmp->GetSkeleton()->getBodyNode("RightShoulder")->getWorldTransform().translation();
	Eigen::Vector3d main_body_dir= (main_ls-main_rs).cross(Eigen::Vector3d::UnitY());

	Eigen::Vector3d main_lf 	= mCharacter_main_tmp->GetSkeleton()->getBodyNode("LeftFoot")->getWorldTransform().translation();
	Eigen::Vector3d main_rf 	= mCharacter_main_tmp->GetSkeleton()->getBodyNode("RightFoot")->getWorldTransform().translation();
	
	main_body_dir = (main_lf-main_rf);
	// main_body_dir /= 2.0;

	Eigen::Vector3d my_root	= mCharacter->GetSkeleton()->getRootBodyNode()->getWorldTransform().translation();

	Eigen::Vector3d my_ls 	= mCharacter->GetSkeleton()->getBodyNode("LeftShoulder")->getWorldTransform().translation();
	Eigen::Vector3d my_rs 	= mCharacter->GetSkeleton()->getBodyNode("RightShoulder")->getWorldTransform().translation();

	Eigen::Vector3d my_body_dir= (my_ls-my_rs).cross(Eigen::Vector3d::UnitY());
	Eigen::Vector3d my_lf 	= mCharacter->GetSkeleton()->getBodyNode("LeftFoot")->getWorldTransform().translation();
	Eigen::Vector3d my_rf 	= mCharacter->GetSkeleton()->getBodyNode("RightFoot")->getWorldTransform().translation();
	
	my_body_dir = (my_lf-my_rf);
	// my_body_dir /= 2.0;

	Eigen::Vector3d look_dir = main_root - my_root;
	Eigen::Vector2d local_coord = DPhy::getLocalCoord_XZ(my_root, my_body_dir, main_root);


	double theta = DPhy::getXZTheta(my_body_dir, look_dir);

	
	if(local_coord.norm() > 1.6){
		mNextMotion = "box_move_front";
		std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	}else if(local_coord.norm() < 1.0){
		mNextMotion = "box_move_back";
		std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	}else if(local_coord[1] > 1.0){
		std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
		mNextMotion = "box_move_left";
	}else if(local_coord[1] < -1.0){
		mNextMotion = "box_move_right";
		std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	}else if(theta > 0.1){
		mNextMotion= "right_pivot_mxm";
		std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	}else if(theta < - 0.3 ){
		mNextMotion = "pivot_mxm";
		std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	}else{
		mNextMotion = "box_idle";
	}

	mCurrentFrameOnPhase++;
	if(mCurrentFrameOnPhase >= mMotionFrames[mCurrentMotion]) mCurrentFrameOnPhase = 0;
	if(mCurrentFrameOnPhase == 0){// && mNextMotion != mCurrentMotion){
		// transition
		mCurrentMotion = mNextMotion;
		mNextMotion= "box_idle";
		std::cout<<mTotalFrame<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
		std::cout<<mTotalFrame<<" / "<<mCurrentMotion<<" / "<<mNextMotion<<std::endl;
		std::cout<<" ======================= "<<mCurrentMotion<<" / "<<mCurrentFrameOnPhase<<" ======================= "<<std::endl;
		calculateAlign();
	}


    Eigen::VectorXd p = mReferenceManager->GetPosition(mCurrentFrameOnPhase, false);
    p = DPhy::MultiplyRootTransform(p, mAlign, true);
	mCharacter->GetSkeleton()->setPositions(p);
	mCharacter->GetSkeleton()->computeForwardKinematics(true, false, false);

	mTotalFrame++;
}

Eigen::VectorXd EnemyKinController::GetPosition(){
	return mCharacter->GetSkeleton()->getPositions();
}

}// end namespace DPhy




// mTransitionRules.emplace(std::make_pair("FW_JUMP", "WALL_JUMP"), std::make_pair(40, 33)); //13
// mTransitionRules.emplace(std::make_pair("WALL_JUMP", "FW_JUMP"), std::make_pair(103, 5)); //13

// mTransitionRules.emplace(std::make_pair("WALL_JUMP", "RUN_SWING"), std::make_pair(92, 23));
// mTransitionRules.emplace(std::make_pair("RUN_SWING", "WALL_JUMP"), std::make_pair(99, 29));

// mTransitionRules.emplace(std::make_pair("FW_JUMP", "RUN_SWING"), std::make_pair(42, 13)); // TODO
// mTransitionRules.emplace(std::make_pair("RUN_SWING", "FW_JUMP"), std::make_pair(95, 0));

// mTransitionRules.emplace(std::make_pair("FW_JUMP", "FW_JUMP"), std::make_pair(60, 0));

