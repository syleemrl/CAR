#include<Eigen/Dense>
#include<iostream>
#include<string>
#include<vector>
#include "Character.h"
#include "ReferenceManager.h"

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

	for(int i=0; i<mReferenceManager->GetPhaseLength(); i++){
		Eigen::VectorXd p = mReferenceManager->GetPosition(i, false);
		ref->GetSkeleton()->setPositions(p);
		ref->GetSkeleton()->computeForwardKinematics(true, false, false);
		Eigen::Vector3d lh= ref->GetSkeleton()->getBodyNode("LeftHand")->getWorldTransform().translation();
		Eigen::Vector3d rh= ref->GetSkeleton()->getBodyNode("RightHand")->getWorldTransform().translation();

		double error= std::min(lh[1], rh[1])-0.035;

		getline(rawfile, raw_line);
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

int main(){
	//stick_hand_to_ground();
	lift_foot_up();
}


