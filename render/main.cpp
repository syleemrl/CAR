#include "SimWindow.h"
#include <vector>
#include <string>
#include <GL/glut.h>

int main(int argc,char** argv)
{
	if( argc < 2 ) {
		std::cout << "Please input a filename" << std::endl;
		return 0;
	}

	std::cout<<"[ : Frame --"<<std::endl;
	std::cout<<"] : Frame ++"<<std::endl;
	std::cout<<"r : Frame = 0"<<std::endl;
	std::cout<<"C : Capture"<<std::endl;
	std::cout<<"SPACE : Play"<<std::endl;
	std::cout<<"ESC : exit"<<std::endl;

	SimWindow* simwindow;
	if( argc == 3 ) {
		std::string network = "";
		simwindow = new SimWindow(std::string(argv[2]), network, std::string(argv[1]));
	}
	else if(argc == 4) {
		simwindow = new SimWindow(std::string(argv[2]), std::string(argv[3]), std::string(argv[1]));
	}
	else {
		simwindow = new SimWindow(std::string(argv[2]), std::string(argv[3]), std::string(argv[1]), std::string(argv[4]));
	}

	// SimWindowMulti* simwindow;
	// std::vector<std::string> network;
	// for(int i = 2; i < argc; i++) {
	// 	network.emplace_back(std::string(argv[i]));
	// }
	// simwindow = new SimWindowMulti(std::string(argv[1]), network);	

	glutInit(&argc, argv);
	simwindow->InitWindow(1600,900,"Render");
	glutMainLoop();
	/*std::vector<int> index;
	index.push_back(0);
	index.push_back(0);

	DPhy::AddZeroToBvh(index, CAR_DIR+"/motion/cartwheel.bvh");*/
	return 0;
}
