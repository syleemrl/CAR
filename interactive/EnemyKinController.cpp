#include "EnemyKinController.h"
#include <tinyxml.h>

namespace DPhy
{

EnemyKinController::EnemyKinController(Eigen::Vector3d pos, Eigen::Vector3d pos_ch)
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

    mReferenceManager = new DPhy::ReferenceManager(new DPhy::Character(path));
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/")+ mCurrentMotion + std::string(".bvh"));

    Eigen::VectorXd p = mReferenceManager->GetPosition(0, true);

    p(3) = pos(0);
    p(5) = pos(2);

	Eigen::Vector3d dir =  pos_ch - p.segment<3>(3);
	
	Eigen::Vector3d ls 	= mCharacter->GetSkeleton()->getBodyNode("LeftShoulder")->getWorldTransform().translation();
	Eigen::Vector3d rs 	= mCharacter->GetSkeleton()->getBodyNode("RightShoulder")->getWorldTransform().translation();
	Eigen::Vector3d my_body_dir= (ls-rs).cross(Eigen::Vector3d::UnitY());

	double theta = DPhy::getXZTheta(my_body_dir, dir);

	Eigen::AngleAxisd root_aa = Eigen::AngleAxisd(p.segment<3>(0).norm(), p.segment<3>(0).normalized());
	Eigen::AngleAxisd rotate_y = Eigen::AngleAxisd(theta, Eigen::Vector3d(0, 1, 0));
	rotate_y = rotate_y * root_aa;
	p.segment<3>(0) = rotate_y.axis() * rotate_y.angle();

	mCharacter->GetSkeleton()->setPositions(p);
	mCharacter->GetSkeleton()->computeForwardKinematics(true, false, false);
	calculateAlign();

    mTotalFrame = 0;
}

void EnemyKinController::Reset(){
	//
}

void EnemyKinController::calculateAlign(){
	Eigen::VectorXd position = mCharacter->GetSkeleton()->getPositions();
	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(position.head<6>());

	Eigen::Isometry3d prev_cycle_end= T_current;
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/")+ mCurrentMotion + std::string(".bvh"));
    Eigen::VectorXd cycle_start_pos = mReferenceManager->GetPosition(mCurrentFrameOnPhase, true);
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
void EnemyKinController::SetAction(std::string action) {
	int blendingInterval = 10;
	mCurrentMotion = action;
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/")+ mCurrentMotion + std::string(".bvh"));

	Eigen::VectorXd pos = mCharacter->GetSkeleton()->getPositions(); // endPosition;
	Eigen::VectorXd pos_not_aligned = mReferenceManager->GetPosition(0, true);
	std::cout << pos.segment<3>(0).transpose() << std::endl;

	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform(pos_not_aligned.head<6>());
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());

	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();
	T01.translation()[1] = 0;

	Eigen::Isometry3d T01_projected = T01;

	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());			
	T01_projected.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));

	Eigen::Isometry3d T0_gen = T01*T0_phase;
	Eigen::Isometry3d T0_gen_projected = T01_projected*T0_phase;
	
	std::vector<Eigen::VectorXd> p;
	std::vector<double> t;
	for(int i = 0; i < mReferenceManager->GetPhaseLength(); i++) {

		Eigen::VectorXd p_tmp = mReferenceManager->GetPosition(i, true);


		Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(p_tmp.head<6>());
		T_current = T0_phase.inverse()*T_current;
		Eigen::Isometry3d T_current_projected = T0_gen_projected*T_current;
		T_current = T0_gen*T_current;
		p_tmp.head<3>() = dart::dynamics::FreeJoint::convertToPositions(T_current_projected).segment<3>(0);
		p_tmp.segment<3>(3) = dart::dynamics::FreeJoint::convertToPositions(T_current).segment<3>(3);

		// p_tmp(4) = mReferenceManager->GetPosition(i, true)(4);

		if(i < blendingInterval-1) {
			double weight = (i+1) / (double)blendingInterval;
			p_tmp = DPhy::BlendPosition(pos, p_tmp, weight, true);
		}
		p.push_back(p_tmp);
		t.push_back(1);

	}

	mReferenceManager->LoadAdaptiveMotion(p, t);
	mCurrentFrameOnPhase = 0;
	mActionSet = true;
}
void EnemyKinController::Step(Eigen::VectorXd main_p){
	if(mPhysicsMode) {
		mCurrentFrameOnPhase++;
		mTotalFrame++;

		return;
	}
	mCharacter_main_tmp->GetSkeleton()->setPositions(main_p);
	mCharacter_main_tmp->GetSkeleton()->computeForwardKinematics(true, false, false);

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
	
	Eigen::AngleAxisd root_temp(mCharacter->GetSkeleton()->getPositions().head<3>().norm(), mCharacter->GetSkeleton()->getPositions().head<3>().normalized());	
	Eigen::Vector3d point_local = look_dir;
	point_local(1) = 0;
	point_local = root_temp.inverse() * look_dir; 
	theta = atan2(point_local(0), point_local(2));

	if(local_coord.norm() > 1.5){
		mNextMotion = "box_move_front";
		//std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	} else if(local_coord.norm() < 0.7){
		mNextMotion = "box_move_back";
		//std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	} else if(local_coord[1] > 1.0){
		//std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
		mNextMotion = "box_move_left";
	} else if(local_coord[1] < -1.0){
		mNextMotion = "box_move_right";
		//std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	} 
	// else if(theta > 0.1){
	// 	mNextMotion= "right_pivot_mxm";
	// 	std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	// }else if(theta < - 0.3 ){
	// 	mNextMotion = "pivot_mxm";
	// 	std::cout<<"@ "<<mTotalFrame<<" / "<<mNextMotion<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
	// }
	else{
		mNextMotion = "box_idle";
	}

	mCurrentFrameOnPhase++;
	if(mCurrentFrameOnPhase >= mMotionFrames[mCurrentMotion]) {
		mActionSet = false;
		mCurrentFrameOnPhase = 0;
	} 
	if(mCurrentFrameOnPhase == 0){// && mNextMotion != mCurrentMotion){
		// transition
		mCurrentMotion = mNextMotion;
		mNextMotion= "box_idle";

		//std::cout<<mTotalFrame<<" // theta : "<<theta<<"/ local_coord: "<<local_coord.transpose()<<std::endl;
		//std::cout<<mTotalFrame<<" / "<<mCurrentMotion<<" / "<<mNextMotion<<std::endl;
		//std::cout<<" ======================= "<<mCurrentMotion<<" / "<<mCurrentFrameOnPhase<<" ======================= "<<std::endl;
		calculateAlign();
	}

    Eigen::VectorXd p = mReferenceManager->GetPosition(mCurrentFrameOnPhase, true);
  	if(mCurrentMotion == "box_idle" && mCurrentFrameOnPhase % 5 == 0 && abs(theta) > 0.3 && local_coord.norm() <1.5) {
  		Eigen::Vector6d root_old = mCharacter->GetSkeleton()->getPositions().head<6>();

		double to_rotate = std::min(0.05, std::max(-0.05, theta));
		Eigen::AngleAxisd root_y_rotate(to_rotate, Eigen::Vector3d(0, 1, 0));

		Eigen::AngleAxisd root_old_aa(root_old.head<3>().norm(), root_old.head<3>().normalized());
		Eigen::AngleAxisd root_new_aa;
		root_new_aa = root_y_rotate * root_old_aa;
		Eigen::Vector3d orientation_new =root_new_aa.axis() * root_new_aa.angle();
		Eigen::Vector6d root_new = root_old;
		root_new.head<3>() = orientation_new;
		Eigen::Isometry3d T_old = dart::dynamics::FreeJoint::convertToTransform(root_old.head<6>());
		Eigen::Isometry3d T_new = dart::dynamics::FreeJoint::convertToTransform(root_new);
   		mAlign =  (T_new * T_old.inverse()) * mAlign;
	}
	// if(mCurrentMotion == "box_idle" && mCurrentFrameOnPhase % 5 == 4 && local_coord.norm() < 1.5) {
	//   	std::cout << local_coord.norm() << std::endl;
	//   	if(local_coord.norm() <= 0.8 || local_coord.norm() >= 1.05) {

	//   		Eigen::Vector6d root_old = mCharacter->GetSkeleton()->getPositions().head<6>();
	//   		Eigen::Vector3d dx;
	//   		if(local_coord.norm() <= 0.8)
	//   			dx << 0, 0, 0.02;
	//   		else
	//   			dx << 0, 0, -0.02;

	// 		Eigen::AngleAxisd root_old_aa(root_old.head<3>().norm(), root_old.head<3>().normalized());
	// 		dx = root_old_aa.inverse() * dx;
	// 		Eigen::Vector6d root_new = root_old;
	// 		root_new.segment<3>(3) += dx;

	// 		Eigen::Isometry3d T_old = dart::dynamics::FreeJoint::convertToTransform(root_old.head<6>());
	// 		Eigen::Isometry3d T_new = dart::dynamics::FreeJoint::convertToTransform(root_new);
	//    		mAlign =  (T_new * T_old.inverse()) * mAlign;
	//   	}
	// }

    if(!mActionSet)
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

