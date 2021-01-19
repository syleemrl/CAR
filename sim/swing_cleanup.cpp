#include<Eigen/Dense>
#include<iostream>
#include<string>
#include<vector>
#include "Character.h"
#include "ReferenceManager.h"
#include "Functions.h"

using namespace std;

 
std::vector<std::string> split(std::string targetStr, std::string token)
{
    // Check parameters
    if(token.length() == 0 || targetStr.find(token) == std::string::npos)
        return std::vector<std::string>({targetStr});
 
    // return var
    std::vector<std::string> ret;
 
    int findOffset  = 0;
    int splitOffset = 0;
    while ((splitOffset = targetStr.find(token, findOffset)) != std::string::npos)
    {
         ret.push_back(targetStr.substr(findOffset, splitOffset - findOffset));
         findOffset = splitOffset + token.length();
    }
    ret.push_back(targetStr.substr(findOffset, targetStr.length() - findOffset));
    
    return ret;
}

std::vector<std::string> split2(std::string input, char delimiter) {
    vector<string> answer;
    stringstream ss(input);
    string temp;
 
    while (getline(ss, temp, delimiter)) {
        answer.push_back(temp);
    }
 
    return answer;
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

//////////////////////////////////////////////// FLAIR cleanup //////////////////////////////////////////////////////////////////
void stick_hand_to_ground(){
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_gen.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);
	mReferenceManager->LoadMotionFromBVH(std::string("/motion/mxm_flair_raw.bvh"));

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+"/motion/mxm_flair_raw.bvh", std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	std::ofstream outfile;
	outfile.open( std::string(CAR_DIR)+"/motion/mxm_flair.bvh", std::ios_base::app); // append instead of overwrite

	for(int f=0; f<mReferenceManager->GetPhaseLength(); f++){
		Eigen::VectorXd p = mReferenceManager->GetPosition(f, false);
		ref->GetSkeleton()->setPositions(p);
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Vector3d lh= ref->GetSkeleton()->getBodyNode("LeftHand")->getWorldTransform().translation();
		Eigen::Vector3d rh= ref->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform().translation();

		double error= std::min(lh[1], rh[1])-0.035;

		getline(rawfile, raw_line); trim(raw_line);
		
		std::vector<std::string> splitted= split(raw_line, " ");
		double new_height = stof(splitted[1])- 100*(error);
		
		std::string newline=splitted[0]+" "+std::to_string(new_height)+" ";
		for(int i=2; i<splitted.size(); i++) newline+= splitted[i]+" ";
		outfile<<newline;
	}
	rawfile.close();
	outfile.close();
}

void lift_foot_up()
{
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_gen.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* refM = new DPhy::ReferenceManager(ref);
	refM->LoadMotionFromBVH(std::string("/motion/mxm_flair.bvh"));

	DPhy::Character* ref_raw = new DPhy::Character(path);
	DPhy::ReferenceManager* refM_raw = new DPhy::ReferenceManager(ref);
	refM_raw->LoadMotionFromBVH(std::string("/motion/mxm_flair_raw.bvh"));

	for(int i=0; i<refM->GetPhaseLength(); i++){
		Eigen::VectorXd p = refM->GetPosition(i, false);
		ref->GetSkeleton()->setPositions(p);
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Vector3d lt= ref->GetSkeleton()->getBodyNode("LeftToe")->getWorldTransform().translation();
		Eigen::Vector3d rt= ref->GetSkeleton()->getBodyNode("RightToe")->getWorldTransform().translation();

		Eigen::VectorXd p_raw = refM_raw->GetPosition(i, false);
		ref_raw->GetSkeleton()->setPositions(p);
		ref_raw->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Vector3d lt_raw= ref_raw->GetSkeleton()->getBodyNode("LeftToe")->getWorldTransform().translation();
		Eigen::Vector3d rt_raw= ref_raw->GetSkeleton()->getBodyNode("RightToe")->getWorldTransform().translation();
		
		//TODO
	}
}

//////////////////////////////////////////////// ROPE SWING cleanup //////////////////////////////////////////////////////////////////

void dart_to_bvh_check()
{

	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_gen.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);
	mReferenceManager->LoadMotionFromBVH(std::string("/motion/swing_start.bvh"));

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+"/motion/swing_start.bvh", std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	for(int i=0; i<20; i++){
		Eigen::VectorXd p_i= mReferenceManager->GetPosition(i, false);
		ref->GetSkeleton()->setPositions(p_i);
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Isometry3d r_i= ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();

		getline(rawfile, raw_line); trim(raw_line);
		
		std::vector<std::string> splitted= split(raw_line, " ");
		Eigen::VectorXd bvh_root(6);

		bvh_root[0]= stof(splitted[3]);
		bvh_root[1]= stof(splitted[4]);
		bvh_root[2]= stof(splitted[5]);

		bvh_root[3]= stof(splitted[0]);
		bvh_root[4]= stof(splitted[1]);
		bvh_root[5]= stof(splitted[2]);
		
		Eigen::VectorXd recon_root(6);
		recon_root= p_i.head<6>();

		Eigen::Vector3d v = recon_root.head<3>();
		double v_n = v.norm();
		v.normalize();
		Eigen::AngleAxisd aa(v_n, v);
		Eigen::Matrix3d m(aa);
		Eigen::Vector3d eulerZXY = dart::math::matrixToEulerZXY(m);

		for(int ri=0; ri<3; ri++) recon_root[ri] = eulerZXY[ri]*180./M_PI; // euler
		for(int ri=3; ri<6; ri++) recon_root[ri] = recon_root[ri]*100;  //translation
		
		std::cout<<i<<std::endl;
		for(int ii=0; ii<6; ii++) std::cout<<bvh_root[ii]<<" | "<<recon_root[ii]<<std::endl;
		std::cout<<std::endl;
			
	}
	rawfile.close();
}

void connect_root()
{

	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_gen.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);
	mReferenceManager->LoadMotionFromBVH(std::string("/motion/swing_merge.bvh"));
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+"/motion/swing_merge.bvh", std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	std::ofstream outfile;
	outfile.open( std::string(CAR_DIR)+"/motion/swing_edit_result.bvh", std::ios_base::app); // append instead of overwrite

	for(int i=0; i<62; i++){
		getline(rawfile, raw_line);
	}

	Eigen::VectorXd p_61= mReferenceManager->GetPosition(61, false);
	ref->GetSkeleton()->setPositions(p_61);
	ref->GetSkeleton()->computeForwardKinematics(true, false, false);
	Eigen::Isometry3d r_61= ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();

	Eigen::VectorXd p_62= mReferenceManager->GetPosition(62, false);
	ref->GetSkeleton()->setPositions(p_62);
	ref->GetSkeleton()->computeForwardKinematics(true, false, false);
	Eigen::Isometry3d r_62= ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();
	Eigen::Isometry3d align = r_61*r_62.inverse();

	for(int i=62; i<mReferenceManager->GetPhaseLength(); i++){
		Eigen::VectorXd p_i= mReferenceManager->GetPosition(i, false);
		ref->GetSkeleton()->setPositions(p_i);
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Isometry3d r_i= ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();
		Eigen::Isometry3d aligned = align* r_i;	

		Eigen::Vector3d aligned_eulerZXY = dart::math::matrixToEulerZXY(aligned.linear());
		aligned_eulerZXY*= 180./M_PI;

		Eigen::Vector3d aligned_p = aligned.translation();
		aligned_p*= 100;

		getline(rawfile, raw_line); trim(raw_line);
		
		std::vector<std::string> splitted= split(raw_line, " ");
		
		std::string newline = "";
		for(int i=0; i<3; i++) newline+= std::to_string(aligned_p(i))+" ";
		for(int i=0; i<3; i++) newline+= std::to_string(aligned_eulerZXY(i))+" ";
		
		for(int i=6; i<splitted.size(); i++) newline+= splitted[i]+" ";
		newline+="\n";
		outfile<<newline;
	}
	rawfile.close();
	outfile.close();
}

void smooth_hand_transition()
{

	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_gen.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);

	std::string raw_file_path = "/motion/swing_edit_result.bvh";
	mReferenceManager->LoadMotionFromBVH(raw_file_path);
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+raw_file_path, std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	std::ofstream outfile;
	std::string outfile_path = "/motion/swing_edit_result_2.bvh";
	outfile.open( std::string(CAR_DIR)+outfile_path, std::ios_base::app); // append instead of overwrite

	for(int i=0; i<62; i++){
		getline(rawfile, raw_line);
	}

	Eigen::VectorXd p_61= mReferenceManager->GetPosition(61, false);
	ref->GetSkeleton()->setPositions(p_61);
	ref->GetSkeleton()->computeForwardKinematics(true, false, false);
	Eigen::Isometry3d r_61= ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();
	Eigen::Isometry3d lh_61= ref->GetSkeleton()->getBodyNode("LeftHand")->getWorldTransform();
	Eigen::Isometry3d rh_61= ref->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform();

	Eigen::VectorXd p_62= mReferenceManager->GetPosition(62, false);
	ref->GetSkeleton()->setPositions(p_62);
	ref->GetSkeleton()->computeForwardKinematics(true, false, false);
	Eigen::Isometry3d r_62= ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();
	Eigen::Isometry3d lh_62= ref->GetSkeleton()->getBodyNode("LeftHand")->getWorldTransform();
	Eigen::Isometry3d rh_62= ref->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform();

	Eigen::Isometry3d align = r_61*r_62.inverse();

	int blend_width = 10;
	for(int i=62-blend_width; i<62+blend_width; i++){

		// IK Clean-up
		Eigen::VectorXd p_i= mReferenceManager->GetPosition(i, false);
		ref->GetSkeleton()->setPositions(p_i);
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Isometry3d r_i= ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();
		Eigen::Isometry3d aligned = align* r_i;	

		Eigen::Vector3d aligned_eulerZXY = dart::math::matrixToEulerZXY(aligned.linear());
		aligned_eulerZXY*= 180./M_PI;

		Eigen::Vector3d aligned_p = aligned.translation();
		aligned_p*= 100;

		getline(rawfile, raw_line); trim(raw_line);
		
		std::vector<std::string> splitted= split(raw_line, " ");
		
		std::string newline = "";
		for(int i=0; i<3; i++) newline+= std::to_string(aligned_p(i))+" ";
		for(int i=0; i<3; i++) newline+= std::to_string(aligned_eulerZXY(i))+" ";
		
		for(int i=6; i<splitted.size(); i++) newline+= splitted[i]+" ";
		newline+="\n";
		outfile<<newline;
	}
	rawfile.close();
	outfile.close();
}

void align_zero_frame(std::string raw_file)
{
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_t3.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);

	std::string raw_file_path = "/motion/"+raw_file+".bvh";
	mReferenceManager->LoadMotionFromBVH(raw_file_path);
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ofstream outfile;
	std::string outfile_path = "/motion/aligned.bvh";
	outfile.open( std::string(CAR_DIR)+outfile_path, std::ios_base::out); 

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+raw_file_path, std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		std::cout<<raw_line<<std::endl;
		outfile<<raw_line<<std::endl;
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	std::vector<Eigen::VectorXd> raw = mReferenceManager->getRawPositions();
	Eigen::VectorXd zero_dir(6); zero_dir.setZero();
	std::vector<Eigen::VectorXd> edited= DPhy::Align(raw, zero_dir) ;

	for(int f=0; f<mReferenceManager->GetPhaseLength(); f++){

		// IK Clean-up
		ref->GetSkeleton()->setPositions(edited[f]);
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Isometry3d edited_root = ref->GetSkeleton()->getBodyNode("Hips")->getWorldTransform();

		Eigen::Vector3d aligned_eulerZXY = dart::math::matrixToEulerZXY(edited_root.linear());
		aligned_eulerZXY*= 180./M_PI;

		Eigen::Vector3d aligned_p = edited_root.translation();
		aligned_p*= 100;

		getline(rawfile, raw_line);
		trim(raw_line);
		std::vector<std::string> splitted= split(raw_line, " ");
		std::cout<<splitted[0]<<" "<<splitted[1]<<" "<<splitted[2]<<std::endl;
		std::cout<<splitted[3]<<" "<<splitted[4]<<" "<<splitted[5]<<std::endl;

		std::string newline = "";
		for(int i=0; i<3; i++) newline+= std::to_string(aligned_p(i))+" ";
		for(int i=0; i<3; i++) newline+= std::to_string(aligned_eulerZXY(i))+" ";
		
		for(int i=6; i<splitted.size(); i++) newline+= splitted[i]+" ";
		newline+="\n";

		outfile<< newline;
	}
	rawfile.close();
	outfile.close();

}

void shift_root(std::string filename, double shift_y){
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_t3.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);

	std::string raw_file_path = "/motion/"+filename+".bvh";
	mReferenceManager->LoadMotionFromBVH(raw_file_path);
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ofstream outfile;
	std::string outfile_path = "/motion/"+filename+"_shifted.bvh";
	outfile.open( std::string(CAR_DIR)+outfile_path, std::ios_base::out); 

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+raw_file_path, std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		std::cout<<raw_line<<std::endl;
		outfile<<raw_line<<std::endl;
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	for(int f=0; f<mReferenceManager->GetPhaseLength(); f++){

		getline(rawfile, raw_line);trim(raw_line);
		
		std::vector<std::string> splitted= split(raw_line, " ");
		double new_height = stof(splitted[1])+ 100*(shift_y);
		
		std::string newline=splitted[0]+" "+std::to_string(new_height)+" ";
		for(int i=2; i<splitted.size(); i++) newline+= splitted[i]+" ";
		newline+="\n";
		outfile<<newline;

	}

	rawfile.close();
	outfile.close();

	std::cout<<outfile_path<<", shift_y: "<<shift_y<<", done"<<std::endl;
}

void stitch_foot_end_to_ground(std::string filename)//, int start, int end)
{
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_t3.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);

	std::string raw_file_path = "/motion/"+filename+".bvh";
	mReferenceManager->LoadMotionFromBVH(raw_file_path);
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ofstream outfile;
	std::string outfile_path = "/motion/"+filename+"_edit.bvh";
	outfile.open( std::string(CAR_DIR)+outfile_path, std::ios_base::out); 

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+raw_file_path, std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		std::cout<<raw_line<<std::endl;
		outfile<<raw_line<<std::endl;
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

		
	for(int frame=0; frame<mReferenceManager->GetPhaseLength(); frame++){
		ref->GetSkeleton()->setPositions(mReferenceManager->GetPosition(frame, false));
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);

		Eigen::Matrix3d lf = ref->GetSkeleton()->getBodyNode("LeftFoot")->getWorldTransform().linear();
		Eigen::Matrix3d rf = ref->GetSkeleton()->getBodyNode("RightFoot")->getWorldTransform().linear();

		Eigen::Matrix3d lf_new= DPhy::projectToXZ(lf);
		Eigen::Matrix3d rf_new= DPhy::projectToXZ(rf);

		Eigen::VectorXd lpos = ref->GetSkeleton()->getBodyNode("LeftFoot")->getParentJoint()->getPositions();
		Eigen::VectorXd rpos = ref->GetSkeleton()->getBodyNode("RightFoot")->getParentJoint()->getPositions();

		Eigen::Matrix3d lRot = dart::dynamics::BallJoint::convertToRotation(lpos);
		Eigen::Matrix3d rRot = dart::dynamics::BallJoint::convertToRotation(rpos);

		lRot = (lf_new*lf.inverse())*lRot;
		rRot = (rf_new*rf.inverse())*rRot;

		ref->GetSkeleton()->getBodyNode("LeftFoot")->getParentJoint()->setPositions(dart::dynamics::BallJoint::convertToPositions(lRot));
		ref->GetSkeleton()->getBodyNode("RightFoot")->getParentJoint()->setPositions(dart::dynamics::BallJoint::convertToPositions(rRot));

		Eigen::VectorXd newpos = ref->GetSkeleton()->getPositions();

		Eigen::Vector3d lf_eulerZXY = dart::math::matrixToEulerZXY(lRot);
		lf_eulerZXY*= 180./M_PI;

		Eigen::Vector3d rf_eulerZXY = dart::math::matrixToEulerZXY(rRot);
		rf_eulerZXY*= 180./M_PI;


		getline(rawfile, raw_line);
		trim(raw_line);
		std::vector<std::string> splitted= split(raw_line, " ");

		// left foot 17 th joint: 3+3*16= 51, 
		// right foot 21 th joint: 3+3*20 =63

		std::string newline = "";		
		for(int i=0; i<51; i++) newline+= splitted[i]+" ";

			newline+= std::to_string(lf_eulerZXY[0])+" "+std::to_string(lf_eulerZXY[1])+" "+std::to_string(lf_eulerZXY[2])+" ";

		for(int i=54; i<63; i++) newline+= splitted[i]+" ";

			newline+= std::to_string(rf_eulerZXY[0])+" "+std::to_string(rf_eulerZXY[1])+" "+std::to_string(rf_eulerZXY[2])+" ";

		for(int i=66; i<splitted.size(); i++) newline+= splitted[i]+" ";

		newline +="\n";
		outfile<<newline;

	}

	rawfile.close();
	outfile.close();

}

void remove_foot_penetration(std::string filename){
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_t3.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);

	std::string raw_file_path = "/motion/"+filename+".bvh";
	mReferenceManager->LoadMotionFromBVH(raw_file_path);
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ofstream outfile;
	std::string outfile_path = "/motion/"+filename+"_rotated.bvh";
	outfile.open( std::string(CAR_DIR)+outfile_path, std::ios_base::out); 

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+raw_file_path, std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		std::cout<<raw_line<<std::endl;
		outfile<<raw_line<<std::endl;
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	int frame = 0;
	ref->GetSkeleton()->setPositions(mReferenceManager->GetPosition(frame, false));
	ref->GetSkeleton()->computeForwardKinematics(true, false, false);

    Eigen::Matrix3d lf_wrot = ref->GetSkeleton()->getBodyNode("LeftFoot")->getWorldTransform().linear();
    Eigen::Matrix3d lf_pRot= lf_wrot* ref->GetSkeleton()->getJoint("LeftFoot")->getRelativeTransform().linear().inverse();
    Eigen::Matrix3d lf_edited= lf_pRot.inverse();

    Eigen::Matrix3d rf_wrot = ref->GetSkeleton()->getBodyNode("RightFoot")->getWorldTransform().linear();
    Eigen::Matrix3d rf_pRot= rf_wrot* ref->GetSkeleton()->getJoint("RightFoot")->getRelativeTransform().linear().inverse();
    Eigen::Matrix3d rf_edited= rf_pRot.inverse();

	Eigen::Matrix3d lf_plus = ref->GetSkeleton()->getJoint("LeftFoot")->getRelativeTransform().linear().inverse()*lf_edited;
	Eigen::Matrix3d rf_plus = ref->GetSkeleton()->getJoint("RightFoot")->getRelativeTransform().linear().inverse()*rf_edited;

	for(int f=0; f<mReferenceManager->GetPhaseLength(); f++){
		ref->GetSkeleton()->setPositions(mReferenceManager->GetPosition(f, false));
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);

		getline(rawfile, raw_line); trim(raw_line);
		std::vector<std::string> splitted= split(raw_line, " ");
			
		Eigen::VectorXd lpos = ref->GetSkeleton()->getBodyNode("LeftFoot")->getParentJoint()->getPositions();
		Eigen::VectorXd rpos = ref->GetSkeleton()->getBodyNode("RightFoot")->getParentJoint()->getPositions();

		Eigen::Matrix3d lRot = dart::dynamics::BallJoint::convertToRotation(lpos);
		Eigen::Matrix3d rRot = dart::dynamics::BallJoint::convertToRotation(rpos);

		lRot = lRot*lf_plus;
		rRot = rRot*rf_plus;

		Eigen::Vector3d lf_eulerZXY = dart::math::matrixToEulerZXY(lRot);
		lf_eulerZXY*= 180./M_PI;

		Eigen::Vector3d rf_eulerZXY = dart::math::matrixToEulerZXY(rRot);
		rf_eulerZXY*= 180./M_PI;


		std::string newline = "";		
		for(int i=0; i<51; i++) newline+= splitted[i]+" ";

			newline+= std::to_string(lf_eulerZXY[0])+" "+std::to_string(lf_eulerZXY[1])+" "+std::to_string(lf_eulerZXY[2])+" ";

		for(int i=54; i<63; i++) newline+= splitted[i]+" ";

			newline+= std::to_string(rf_eulerZXY[0])+" "+std::to_string(rf_eulerZXY[1])+" "+std::to_string(rf_eulerZXY[2])+" ";

		for(int i=66; i<splitted.size(); i++) newline+= splitted[i]+" ";

		newline+="\n";
		outfile<<newline;

	}

	rawfile.close();
	outfile.close();
	std::cout<<outfile_path<<" DONE"<<std::endl;
}


void stitch_toe(std::string filename){
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_t3.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);

	std::string raw_file_path = "/motion/"+filename+".bvh";
	mReferenceManager->LoadMotionFromBVH(raw_file_path);
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ofstream outfile;
	std::string outfile_path = "/motion/"+filename+"_stitch_toe.bvh";
	outfile.open( std::string(CAR_DIR)+outfile_path, std::ios_base::out); 

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+raw_file_path, std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		std::cout<<raw_line<<std::endl;
		outfile<<raw_line<<std::endl;
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	for(int f=0; f<mReferenceManager->GetPhaseLength(); f++){
		ref->GetSkeleton()->setPositions(mReferenceManager->GetPosition(f, false));
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);

		if(f>=57 && f<=71){
			getline(rawfile, raw_line); 
			outfile<<raw_line;
		}else{

			getline(rawfile, raw_line); trim(raw_line);
			std::vector<std::string> splitted= split(raw_line, " ");

		    Eigen::Matrix3d lf_wrot = ref->GetSkeleton()->getBodyNode("LeftToe")->getWorldTransform().linear();
		    Eigen::Matrix3d lf_pRot= lf_wrot* ref->GetSkeleton()->getJoint("LeftToe")->getRelativeTransform().linear().inverse();
		    Eigen::Matrix3d lf_edited= lf_pRot.inverse();

		    Eigen::Matrix3d rf_wrot = ref->GetSkeleton()->getBodyNode("RightToe")->getWorldTransform().linear();
		    Eigen::Matrix3d rf_pRot= rf_wrot* ref->GetSkeleton()->getJoint("RightToe")->getRelativeTransform().linear().inverse();
		    Eigen::Matrix3d rf_edited= rf_pRot.inverse();
				
			Eigen::Vector3d lf_eulerZXY = dart::math::matrixToEulerZXY(lf_edited);
			lf_eulerZXY*= 180./M_PI;

			Eigen::Vector3d rf_eulerZXY = dart::math::matrixToEulerZXY(rf_edited);
			rf_eulerZXY*= 180./M_PI;

			std::string newline = "";		
			for(int i=0; i<54; i++) newline+= splitted[i]+" ";

				newline+= std::to_string(lf_eulerZXY[0])+" "+std::to_string(lf_eulerZXY[1])+" "+std::to_string(lf_eulerZXY[2])+" ";

			for(int i=57; i<66; i++) newline+= splitted[i]+" ";

				newline+= std::to_string(rf_eulerZXY[0])+" "+std::to_string(rf_eulerZXY[1])+" "+std::to_string(rf_eulerZXY[2])+" ";

			for(int i=69; i<splitted.size(); i++) newline+= splitted[i]+" ";

			newline+="\n";
			outfile<<newline;
		}
	}

	rawfile.close();
	outfile.close();
	std::cout<<outfile_path<<" DONE"<<std::endl;
}

void parse_cleanup(std::string filename)
{
	std::string path = std::string(CAR_DIR)+std::string("/character/mxm_t3.xml");
	DPhy::Character* ref = new DPhy::Character(path);
	DPhy::ReferenceManager* mReferenceManager = new DPhy::ReferenceManager(ref);

	std::string raw_file_path = "/motion/"+filename+".bvh";
	mReferenceManager->LoadMotionFromBVH(raw_file_path);
	std::cout<<"total frame :"<<mReferenceManager->GetPhaseLength()<<std::endl;

	std::ofstream outfile;
	std::string outfile_path = "/motion/"+filename+"_tmp.bvh";
	outfile.open( std::string(CAR_DIR)+outfile_path, std::ios_base::out); 

	std::ifstream rawfile;
	rawfile.open(std::string(CAR_DIR)+raw_file_path, std::ios_base::in); 	
	std::string raw_line;
	while(true){
		getline(rawfile, raw_line);
		std::cout<<raw_line<<std::endl;
		outfile<<raw_line<<std::endl;
		if(raw_line.find("Time:")!=std::string::npos){
			break;
		}
	}

	char buffer[256];
	double val;

	for(int f=0; f<mReferenceManager->GetPhaseLength(); f++){
		for(int i=0; i<69; i++) {
			rawfile>> val;
			outfile<<val<<" ";
		}
		outfile<<"\n";
	}
	rawfile.close();
	outfile.close();
}
int main(int argc, char ** argv){
	
	// FLAIR cleanup
	//stick_hand_to_ground();
	// lift_foot_up();

	// SWING cleanup
	// dart_to_bvh_check();
	// connect_root();	

	std::string command = argv[1];

	if(command.compare("align") == 0) align_zero_frame(argv[2]);
	else if(command.compare("foot_end") == 0)stitch_foot_end_to_ground(argv[2]);
	else if(command.compare("shift_root") == 0) shift_root(argv[2], std::stof(argv[3]));
	else if(command.compare("rotate_foot") == 0) remove_foot_penetration(argv[2]);
	else if(command.compare("stitch_toe") == 0) stitch_toe(argv[2]);
	else if(command.compare("parse_cleanup") == 0) parse_cleanup(argv[2]);
	
	else std::cout<<"NO COMMAND FOUND"<<std::endl;
}

