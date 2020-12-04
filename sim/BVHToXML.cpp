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

	void BVHToXML(BVHNode* node, TiXmlElement *xml, Eigen::Vector3d offset, std::vector<TiXmlElement*> &list, TiXmlElement *general_doc){
		TiXmlElement* body = new TiXmlElement("BodyPosition");
		list.push_back(xml);

		xml->LinkEndChild(body);

		TiXmlElement* joint = new TiXmlElement("JointPosition");
		xml->LinkEndChild(joint);
		joint->SetAttribute("translation", v3toString(offset));
		
		for(BVHNode* c : node->child){
			TiXmlElement* child = new TiXmlElement("Joint");
			child->SetAttribute("type", "BallJoint");
			child->SetAttribute("parent_name", node->name);
			child->SetAttribute("size", "0.1 0.1 0.1");
			child->SetAttribute("name", c->name);
			child->SetAttribute("bvh", c->name);

			if(Eigen::Vector3d(c->offset).norm() >= 1e-6){
				if (node->name == "Hips") {
					if(c->name =="Spine") {
						xml->SetAttribute("size", v3toString(Eigen::Vector3d(0.2, 2*Eigen::Vector3d(c->offset).norm(), 0.13)));
						body->SetAttribute("translation", v3toString(Eigen::Vector3d::Zero()));
					}
				}else if ((node->name.find("Arm") != std::string::npos) || (node->name.find("Hand") != std::string::npos) ){
					xml->SetAttribute("size", v3toString(Eigen::Vector3d(Eigen::Vector3d(c->offset).norm(), 0.07, 0.07)));	
					body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); 
				}else if(node->name=="Spine2"){
					if(c->name =="Neck") {
						xml->SetAttribute("size", v3toString(Eigen::Vector3d(0.1, Eigen::Vector3d(c->offset).norm(), 0.13)));
						body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); //body == draw
					}
				}else{					
					if(node->name.find("Spine")!= std::string::npos) xml->SetAttribute("size", v3toString(Eigen::Vector3d(0.2, Eigen::Vector3d(c->offset).norm(), 0.13)));
					else if(node->name.find("Head")!= std::string::npos) xml->SetAttribute("size", v3toString(Eigen::Vector3d(0.11, 0.7*Eigen::Vector3d(c->offset).norm(), 0.11)));
					else if(node->name.find("Neck")!= std::string::npos) xml->SetAttribute("size", v3toString(Eigen::Vector3d(0.07, Eigen::Vector3d(c->offset).norm(), 0.07)));
					else {
						std::cout<<node->name<<std::endl;
						xml->SetAttribute("size", v3toString(Eigen::Vector3d(0.1, Eigen::Vector3d(c->offset).norm(), 0.1)));
					}
					body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); //body == draw
				}

				// if (node->name == "Hips") {
				// 	if(c->name =="Spine") {
				// 		child->SetAttribute("size", v3toString(Eigen::Vector3d(0.2, 2*Eigen::Vector3d(c->offset).norm(), 0.1)));
				// 		body->SetAttribute("translation", v3toString(Eigen::Vector3d::Zero()));
				// 	}
				// }else if ((node->name.find("Arm") != std::string::npos) || (node->name.find("Hand") != std::string::npos) ){
				// 	child->SetAttribute("size", v3toString(Eigen::Vector3d(Eigen::Vector3d(c->offset).norm(), 0.07, 0.07)));	
				// 	body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); 
				// }else if(node->name=="Spine2"){
				// 	if(c->name =="Neck") {
				// 		child->SetAttribute("size", v3toString(Eigen::Vector3d(Eigen::Vector3d(c->offset).norm(), 0.1, 0.1)));
				// 		body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); //body == draw
				// 	}
				// }else if(c->name.find("Toe")!= std::string::npos){
				// 	// Foot->Toe
				// 	Eigen::Vector3d foot_position = offset+ Eigen::Vector3d(c->offset)/2;
				// 	double foot_length = Eigen::Vector3d(c->offset).norm();
				// 	foot_height = default_height + foot_position[1] ;
				// 	std::cout<<"foot_height : "<<foot_height<<std::endl;
				// 	double foot_front = sqrt(foot_length*foot_length - foot_height*foot_height);

				// 	child->SetAttribute("size", v3toString(Eigen::Vector3d( 0.08, 2*foot_height,foot_front)));
				// 	body->SetAttribute("translation",v3toString(foot_position)); //body == draw
				// }else if(node->name.find("Toe")!= std::string::npos){
				// 	// Toe->ToeEnd
				// 	child->SetAttribute("size", v3toString(Eigen::Vector3d(0.08, 2*foot_height, Eigen::Vector3d(c->offset).norm())));
					
				// 	Eigen::Vector3d toe_position = offset+ Eigen::Vector3d(c->offset)/2;
				// 	toe_position[1] = foot_height-default_height;
				// 	body->SetAttribute("translation",v3toString(toe_position)); //body == draw
				// }else{
				// 	if(node->name.find("Spine")!= std::string::npos) child->SetAttribute("size", v3toString(Eigen::Vector3d(0.2, 0.1, Eigen::Vector3d(c->offset).norm())));
				// 	else if(node->name.find("Head")!= std::string::npos) child->SetAttribute("size", v3toString(Eigen::Vector3d(0.16, Eigen::Vector3d(c->offset).norm(), 0.16)));
				// 	else if(node->name.find("Leg")!= std::string::npos) child->SetAttribute("size", v3toString(Eigen::Vector3d(Eigen::Vector3d(c->offset).norm(), 0.16, 0.16)));
				// 	else child->SetAttribute("size", v3toString(Eigen::Vector3d(0.1, Eigen::Vector3d(c->offset).norm(), 0.1)));
					
				// 	body->SetAttribute("translation",v3toString(offset+ Eigen::Vector3d(c->offset)/2)); //body == draw
				// }
			}

			if(c->name == "Site") continue;
			if(c->name == "Head") BVHToXML(c, child, offset + 0.7*Eigen::Vector3d(c->offset), list, general_doc);
			else BVHToXML(c, child, offset + Eigen::Vector3d(c->offset), list, general_doc);
		}


		if(general_doc->FirstChildElement(node->name)!= nullptr) {
			TiXmlElement* elm= general_doc->FirstChildElement(node->name);
			xml->SetAttribute("mass", elm->Attribute("mass")); 
			if(elm->Attribute("type")!= nullptr){
				xml->SetAttribute("type", elm->Attribute("type"));		
			}
			if(elm->Attribute("name")!= nullptr){
				xml->SetAttribute("name", elm->Attribute("name"));		
			}
			if(elm->Attribute("bvh")!= nullptr){
				xml->SetAttribute("bvh", elm->Attribute("bvh"));
			}
			if(elm->Attribute("TorqueLimit")!= nullptr){
				TiXmlElement* TorqueLimit = new TiXmlElement("TorqueLimit");
				xml->LinkEndChild(TorqueLimit);
				TorqueLimit->SetAttribute("norm", general_doc->FirstChildElement(node->name)->Attribute("TorqueLimit"));
			}
		}

		if(body->Attribute("translation")== nullptr) body->SetAttribute("translation",v3toString(offset));
	}

	void BVHToFile(BVHNode* root, std::string filename, TiXmlElement* general_doc){
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

		if(general_doc->FirstChildElement(root->name)!= nullptr) {
			TiXmlElement* child_elm = general_doc->FirstChildElement(root->name);
			child->SetAttribute("mass", child_elm->Attribute("mass")); 
			if(child_elm->Attribute("type")!= nullptr){
				child->SetAttribute("type", child_elm->Attribute("type"));		
			}
			if(child_elm->Attribute("name")!= nullptr){
				child->SetAttribute("name", child_elm->Attribute("name"));		
			}
			if(child_elm->Attribute("TorqueLimit")!= nullptr){
				TiXmlElement* TorqueLimit = new TiXmlElement("TorqueLimit");
				child->LinkEndChild(TorqueLimit);
				TorqueLimit->SetAttribute("norm", general_doc->FirstChildElement(root->name)->Attribute("TorqueLimit"));
			}


		}


		BVHToXML(root, child, Eigen::Vector3d::Zero(), /*(root->offset),*/ list, general_doc);
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
	// TiXmlDocument* general_doc = new TiXmlDocument("../character/humanoid_general.xml");
	std::string filename= "../character/humanoid_general.xml";
	TiXmlDocument general_doc;
	if(!general_doc.LoadFile(filename)){
		std::cout << "Can't open file : " << filename << std::endl;
		// return nullptr;
	}

	TiXmlElement * skeldoc = general_doc.FirstChildElement("Skeleton");

	myBVH::BVHToFile(myBVH::BVHParser(argv[1]), argv[2], skeldoc);
}