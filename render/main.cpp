#include "SimWindow.h"
#include "KinematicMotionWindow.h"
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

	std::string type = std::string(argv[1]);
	if(type.compare("sim")==0) {
		SimWindow* simwindow;
		if( argc == 4 ) {
			std::string network = "";
			simwindow = new SimWindow(std::string(argv[3]), network, std::string(argv[2]));
		}
		else if(argc == 5) {
			simwindow = new SimWindow(std::string(argv[3]), std::string(argv[4]), std::string(argv[2]));
		}
		else {
			simwindow = new SimWindow(std::string(argv[3]), std::string(argv[4]), std::string(argv[2]), std::string(argv[5]));
		}
		glutInit(&argc, argv);
		simwindow->InitWindow(1600,900,"Render");
		glutMainLoop();
	}
	else {
		int n = atoi(argv[2]);
		std::vector<std::string> motion;
		std::vector<std::string> mode;
		for(int i = 0; i < n; i++) {
			mode.push_back(std::string(argv[3+i*2]));
			motion.push_back(std::string(argv[3+i*2+1]));
		}
		KinematicMotionWindow* kwindow = new KinematicMotionWindow(motion, mode);

		glutInit(&argc, argv);
		kwindow->InitWindow(1600,900,"Render");
		glutMainLoop();
	}

	return 0;
}
