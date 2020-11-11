#include<tinyxml.h>
#include<Eigen/Dense>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>


std::string v3toString(Eigen::Vector3d vec){
	std::stringstream ss; ss << vec[0] << " " << vec[1] << " " << vec[2] << " ";
	return ss.str();
}

double default_height = 0;
double foot_height = 0;

namespace myBVH{
	struct BVHNode{
		std::string name;
		double offset[3];
		std::vector<std::string> channelList;
		std::vector<BVHNode*> child;
	};

	void BVHToXML(BVHNode* node, TiXmlElement *xml, Eigen::Vector3d offset, std::vector<TiXmlElement*> &list){
		TiXmlElement* body = new TiXmlElement("BodyPosition");
		list.push_back(xml);

		xml->LinkEndChild(body);
		// body->SetAttribute("linear", "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 ");
		// body->SetAttribute("translation",v3toString(offset)); //PFNN_GEN: body position = joint position

		TiXmlElement* joint = new TiXmlElement("JointPosition");
		xml->LinkEndChild(joint);
		joint->SetAttribute("translation", v3toString(offset));
		
		for(BVHNode* c : node->child){
			TiXmlElement* child = new TiXmlElement("Joint");
			child->SetAttribute("type", "BallJoint");
			child->SetAttribute("name", c->name);
			child->SetAttribute("parent_name", node->name);
			child->SetAttribute("size", "0.1 0.1 0.1");
			child->SetAttribute("mass", "1");
			child->SetAttribute("bvh", c->name);

			if(Eigen::Vector3d(c->offset).norm() >= 1e-6){
				if (node->name == "Hips") {
					if(c->name =="Spine") {
						TiXmlElement* capsule = new TiXmlElement("Box");
						capsule->SetAttribute("offset", v3toString(Eigen::Vector3d::Zero()));  // body == draw
						capsule->SetAttribute("direction", v3toString(Eigen::Vector3d(c->offset).normalized()));
						capsule->SetAttribute("size", v3toString(Eigen::Vector3d(0.2, 2*Eigen::Vector3d(c->offset).norm(), 0.1)));
						body->SetAttribute("translation", v3toString(Eigen::Vector3d::Zero()));
						xml->LinkEndChild(capsule);
					}
				}else if ((node->name.find("Arm") != std::string::npos) || (node->name.find("Hand") != std::string::npos) ){
					TiXmlElement* capsule = new TiXmlElement("Box");
					capsule->SetAttribute("offset", v3toString(Eigen::Vector3d::Zero()));  // body == draw
					capsule->SetAttribute("direction", v3toString(Eigen::Vector3d(c->offset).normalized()));
					capsule->SetAttribute("size", v3toString(Eigen::Vector3d(Eigen::Vector3d(c->offset).norm(), 0.07, 0.07)));	
					body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); 
					xml->LinkEndChild(capsule);

				}else if(node->name=="Spine2"){
					if(c->name =="Neck") {
						TiXmlElement* capsule = new TiXmlElement("Box");
						capsule->SetAttribute("offset", v3toString(Eigen::Vector3d::Zero()));  // body == draw
						capsule->SetAttribute("direction", v3toString(Eigen::Vector3d(c->offset).normalized()));
						capsule->SetAttribute("size", v3toString(Eigen::Vector3d(0.1, Eigen::Vector3d(c->offset).norm(), 0.1)));
						body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); //body == draw
						xml->LinkEndChild(capsule);
					}
				}else if(c->name.find("Toe")!= std::string::npos){
					// Foot->Toe
					TiXmlElement* capsule = new TiXmlElement("Box");
					capsule->SetAttribute("offset", v3toString(Eigen::Vector3d::Zero()));  // body == draw
					capsule->SetAttribute("direction", v3toString(Eigen::Vector3d(c->offset).normalized()));
					
					Eigen::Vector3d foot_position = offset+ Eigen::Vector3d(c->offset)/2;
					double foot_length = Eigen::Vector3d(c->offset).norm();
					foot_height = default_height + foot_position[1] ;
					std::cout<<"foot_height : "<<foot_height<<std::endl;
					double foot_front = sqrt(foot_length*foot_length - foot_height*foot_height);

					capsule->SetAttribute("size", v3toString(Eigen::Vector3d( 0.08, 2*foot_height,foot_front)));
					

					body->SetAttribute("translation",v3toString(foot_position)); //body == draw
					xml->LinkEndChild(capsule);

				}else if(node->name.find("Toe")!= std::string::npos){
					// Toe->ToeEnd
					TiXmlElement* capsule = new TiXmlElement("Box");
					capsule->SetAttribute("offset", v3toString(Eigen::Vector3d::Zero()));  // body == draw
					capsule->SetAttribute("direction", v3toString(Eigen::Vector3d(c->offset).normalized()));
					
					capsule->SetAttribute("size", v3toString(Eigen::Vector3d(0.08, 2*foot_height, Eigen::Vector3d(c->offset).norm())));
					
					Eigen::Vector3d toe_position = offset+ Eigen::Vector3d(c->offset)/2;
					toe_position[1] = foot_height-default_height;
					body->SetAttribute("translation",v3toString(toe_position)); //body == draw
					xml->LinkEndChild(capsule);

				}else{
					TiXmlElement* capsule = new TiXmlElement("Box");
					capsule->SetAttribute("offset", v3toString(Eigen::Vector3d::Zero()));  // body == draw
					capsule->SetAttribute("direction", v3toString(Eigen::Vector3d(c->offset).normalized()));
					
					if(node->name.find("Spine")!= std::string::npos) capsule->SetAttribute("size", v3toString(Eigen::Vector3d(0.2, Eigen::Vector3d(c->offset).norm(), 0.1)));
					else if(node->name.find("Head")!= std::string::npos) capsule->SetAttribute("size", v3toString(Eigen::Vector3d(0.16, Eigen::Vector3d(c->offset).norm(), 0.16)));
					else capsule->SetAttribute("size", v3toString(Eigen::Vector3d(0.1, Eigen::Vector3d(c->offset).norm(), 0.1)));
					
					body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); //body == draw
					xml->LinkEndChild(capsule);

					// if(node->name.find("Foot")!=std::string::npos || node->name.find("Toe")!=std::string::npos
					// 	||c->name.find("Foot")!=std::string::npos || c->name.find("Toe")!=std::string::npos) {
					// 	std::cout<<node->name<<" "<<c->name<<"\t"<<offset.transpose()<<"\t"<<(offset.norm())<<" "<<Eigen::Vector3d(c->offset).norm()<<std::endl;
					// }
				}
				// xml->SetAttribute("mass", std::to_string(std::pow(Eigen::Vector3d(c->offset).norm()*4, 3))); 
			}

			if(c->name == "Site") continue;
			BVHToXML(c, child, offset + Eigen::Vector3d(c->offset), list);
		}

		TiXmlElement* TorqueLimit = new TiXmlElement("TorqueLimit");
		xml->LinkEndChild(TorqueLimit);
		TorqueLimit->SetAttribute("norm", "100");

		if(body->Attribute("translation")== nullptr) body->SetAttribute("translation",v3toString(offset));
	}

	void BVHToFile(BVHNode* root, std::string filename){
		TiXmlDocument* doc = new TiXmlDocument(filename);
		std::vector<TiXmlElement*> list;

		TiXmlElement* skel = new TiXmlElement("Skeleton");
		doc->LinkEndChild(skel);
		skel->SetAttribute("name", "Humanoid");

		TiXmlElement* child = new TiXmlElement("Joint");
		child->SetAttribute("type", "FreeJoint");
		child->SetAttribute("name", root->name);
		child->SetAttribute("parent_name", "None");
		child->SetAttribute("size", "0.05 0.05 0.05");
		child->SetAttribute("mass", "1"); // interactive if want to
		child->SetAttribute("bvh", root->name);

		BVHToXML(root, child, Eigen::Vector3d::Zero(), /*(root->offset),*/ list);
		for(auto elem : list){
			skel->LinkEndChild(elem);
		}

		doc->SaveFile();
	}

	BVHNode* endParser(std::ifstream &file){
		BVHNode* current_node = new BVHNode();
		std::string tmp;
		file >> current_node->name;
		file >> tmp; assert(tmp == "{");
		file >> tmp; assert(tmp == "OFFSET");
		for(int i = 0; i < 3; i++) {
			double d;
			file >> d; 
			current_node->offset[i] = d/100;
		}
		file >> tmp; assert(tmp == "}");

		return current_node;
	}

	BVHNode* nodeParser(std::ifstream &file)
	{
		std::string command, tmp;
		int sz;
		BVHNode* current_node = new BVHNode();

		file >> current_node->name;
		file >> tmp; assert(tmp == "{");

		while(1){
			file >> command;
			if(command == "OFFSET"){
				for(int i = 0; i < 3; i++){
					file >> current_node->offset[i];
					current_node->offset[i]/= 100;
				}
			}
			else if(command == "CHANNELS"){
				file >> sz;
				for(int i = 0; i < sz; i++){
					file >> tmp;
					current_node->channelList.push_back(tmp);
				}
			}
			else if(command == "JOINT"){
				current_node->child.push_back(nodeParser(file));
			}
			else if(command == "End"){
				current_node->child.push_back(endParser(file));
			}
			else if(command == "}") break;
		}
		return current_node;
	}
	BVHNode* BVHParser(std::string filename)
	{
		std::ifstream file(filename);
		std::string command, tmp;

		file >> command; assert(command == "HIERARCHY");
		file >> tmp; assert(tmp == "ROOT");
		BVHNode *root = nodeParser(file);

		for(int i=0; i<8; i++) {
			// MOTION / Frames: / 30/ Frame/ Time: / 0.03333/ x / y
			file >> tmp; 
		}
		std::cout<<tmp<<std::endl;
		default_height = std::stof(tmp)/100;

		file.close();

		return root;
	}
}

int main(int argc, char** argv)
{
	myBVH::BVHToFile(myBVH::BVHParser(argv[1]), argv[2]);
}