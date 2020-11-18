
// MOTION WIDGET : 

// to calculate how to move root to stick the foot (from naive(foot sliding) standing bvh)
	for(int i=35; i<42; i++)
	{
		mSkel_bvh->setPositions(mMotion_bvh[i]);
		mSkel_bvh->computeForwardKinematics(true, false, false);
		Eigen::Vector3d lf = mSkel_bvh->getBodyNode("LeftFoot")->getWorldTransform().translation();
		Eigen::Vector3d rf = mSkel_bvh->getBodyNode("RightFoot")->getWorldTransform().translation();
		Eigen::Vector3d middle= (lf+rf)/2;
		middle[0]+= 0.75;
		std::cout<<(i)<<" "<<middle.transpose()<<std::endl;//<<foot.transpose()<<"\t/ "<<root.transpose()<<"\t/"<<new_root.transpose()<<std::endl;
	}

	// 35 0.020062 0.531804 0.265839
	// 36 0.0176278   0.59416   0.35295
	// 37 0.0110007  0.624677  0.453007
	// 38 0.00593657   0.630926   0.551213
	// 39 0.00503803   0.618199   0.633275
	// 40 0.00724252   0.587761   0.690165
	// 41 0.0104028  0.547423  0.719404
	// 0, hips: -0.750086   1.04059  0.016015

// to calculate foot distance (start(frame 0) ~ stabel stand (frame 50))

	mSkel_bvh->setPositions(mMotion_bvh[0]);
	mSkel_bvh->computeForwardKinematics(true, false, false);
	Eigen::Vector3d root = mSkel_bvh->getBodyNode("Hips")->getWorldTransform().translation();
	Eigen::Vector3d lf = mSkel_bvh->getBodyNode("LeftFoot")->getWorldTransform().translation();
	Eigen::Vector3d rf = mSkel_bvh->getBodyNode("RightFoot")->getWorldTransform().translation();
	Eigen::Vector3d middle= (lf+rf)/2;
	std::cout<<"0, hips: "<<root.transpose()<<std::endl; 
	std::cout<<"0, middle: "<<middle.transpose()<<std::endl; 

	mSkel_bvh->setPositions(mMotion_bvh[50]);
	mSkel_bvh->computeForwardKinematics(true, false, false);
	lf = mSkel_bvh->getBodyNode("LeftFoot")->getWorldTransform().translation();
	rf = mSkel_bvh->getBodyNode("RightFoot")->getWorldTransform().translation();
	middle= (lf+rf)/2;
	std::cout<<"50, middle: "<<middle.transpose()<<std::endl; 

	// 0, middle: -0.739319 0.0438206 0.0566185
	// 45, middle: -0.739077  0.495829  0.710883
	// 50, middle: -0.740508  0.502092  0.718014

	

	// testing prismatic joint
	Eigen::VectorXd obj_pos(mSkel_tmp->getNumDofs());
	obj_pos.setZero();
	obj_pos[5] = -1;
	obj_pos[6] = 1;
	mSkel_tmp->setPositions(obj_pos);
	mSkel_tmp->setVelocities(Eigen::VectorXd::Zero(mSkel_tmp->getNumDofs()));
	mSkel_tmp->setAccelerations(Eigen::VectorXd::Zero(mSkel_tmp->getNumDofs()));
	mSkel_tmp->computeForwardKinematics(true,false,false);
