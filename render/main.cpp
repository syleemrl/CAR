#include "SimWindow.h"
#include "RecordWindow.h"
#include <vector>
#include <string>
#include <GL/glut.h>

int main(int argc,char** argv)
{

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
		std::vector<std::string> motion;
		for(int i = 2; i < argc; i++) {
			motion.push_back(std::string(argv[i]));
		}
		RecordWindow* rwindow = new RecordWindow(motion);

		glutInit(&argc, argv);
		rwindow->InitWindow(1600,900,"Render");
		glutMainLoop();
	}

	return 0;
}
