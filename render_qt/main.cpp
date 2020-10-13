#include <vector>
#include <string>
#include <GL/glut.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <algorithm>
#include <regex>
#include <QApplication>
#include <QGLWidget>
#include "MainWindow.h"
class GLWidget : public QGLWidget{
    void initializeGL(){
        glClearColor(0.0, 1.0, 1.0, 1.0);
    }
    
    void qgluPerspective(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar){
        const GLdouble ymax = zNear * tan(fovy * M_PI / 360.0);
        const GLdouble ymin = -ymax;
        const GLdouble xmin = ymin * aspect;
        const GLdouble xmax = ymax * aspect;
        glFrustum(xmin, xmax, ymin, ymax, zNear, zFar);
    }
    
    void resizeGL(int width, int height){
        if (height==0) height=1;
        glViewport(0,0,width,height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        qgluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
     }
    
    void paintGL(){
        glMatrixMode(GL_MODELVIEW);         
        glLoadIdentity();
        glClear(GL_COLOR_BUFFER_BIT);  

        glBegin(GL_POLYGON); 
            glVertex2f(-0.5, -0.5); 
            glVertex2f(-0.5, 0.5);
            glVertex2f(0.5, 0.5); 
            glVertex2f(0.5, -0.5); 
        glEnd();
    }
};
bool compare_string_with_number(const std::string& s1, const std::string& s2) {
	int length = std::min(s1.length(), s2.length());
	for(int i = 0; i < length; i++) {
		if(s1.at(i) >= 48 && s1.at(i) <= 57 && s2.at(i) >= 48 && s2.at(i) <= 57) {
			int n1 = 0, n2 = 0;
			n1 = (int)s1.at(i) - 48;
			n2 = (int)s2.at(i) - 48;

			int l1 = 1, l2 = 1;
			while(s1.length() > i + l1 && s1.at(i + l1) >= 48 && s1.at(i + l1) <= 57 ) {
				n1 *= 10;
				n1 += (int)s1.at(i + l1) - 48;
				l1 += 1;
			}

			while(s2.length() > i + l2 &&  s2.at(i + l2) >= 48 && s2.at(i + l2) <= 57 ) {
				n2 *= 10;
				n2 += (int)s2.at(i + l2) - 48;
				l2 += 1;
			}

			if(n1 < n2) {
				return true;
			} else if(n1 > n2) {
				return false;
			}

			i += (l1 - 1);
		}
		else {
			if(s1.at(i) < s2.at(i)) {
				return true;
			} else if (s1.at(i) > s2.at(i)) {
				return true;
			}
		} 
	}
	return false;
}
int main(int argc,char** argv)
{
	std::cout<<"[ : Frame --"<<std::endl;
	std::cout<<"] : Frame ++"<<std::endl;
	std::cout<<"r : Frame = 0"<<std::endl;
	std::cout<<"C : Capture"<<std::endl;
	std::cout<<"SPACE : Play"<<std::endl;
	std::cout<<"ESC : exit"<<std::endl;

	boost::program_options::options_description desc("allowed options");
	desc.add_options()
	("type,t",boost::program_options::value<std::string>())
	("file,f",boost::program_options::value<std::string>())
	("ref,r",boost::program_options::value<std::string>())
	("network,n",boost::program_options::value<std::string>())
	("save,s",boost::program_options::value<std::string>())
	;

	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	std::string type, file="", network="", save="", ref="";

	if(vm.count("type")) {
		type = vm["type"].as<std::string>();
	}
	if(vm.count("file")) {
		file = vm["file"].as<std::string>();
	}
	if(vm.count("network")) {
		network = vm["network"].as<std::string>();
	}
	if(vm.count("save")) {
		save = vm["save"].as<std::string>();
	}
	if(vm.count("ref")) {
		ref = vm["ref"].as<std::string>();
	}

	// if(type.compare("sim")==0) {
	// 	SimWindow* simwindow = new SimWindow(ref, network, save);
	// 	glutInit(&argc, argv);
	// 	simwindow->InitWindow(1600,900,"Render");
	// 	glutMainLoop();
	// } else if(type.compare("spl_to_pos")==0) {
	// 	SplineWindow* swindow = new SplineWindow(ref, std::string(CAR_DIR) + std::string("/") + file, "spline");
	// 	glutInit(&argc, argv);
	// 	swindow->InitWindow(1600,900,"Render");
	// 	glutMainLoop();
	// } else if(type.compare("pos_to_spl")==0) {
	// 	SplineWindow* swindow = new SplineWindow(ref, std::string(CAR_DIR) + std::string("/") + file, "position");
	// 	glutInit(&argc, argv);
	// 	swindow->InitWindow(1600,900,"Render");
	// 	glutMainLoop();
	// } else if(type.compare("rec")==0 || type.compare("srec")==0){
	// 	std::vector<std::string> raw = DPhy::split(file, ',');
	// 	std::vector<std::string> motion;

	// 	for(int i = 0; i < raw.size(); i++) {	
	// 		std::vector<std::string> path;
	// 		path.push_back(std::string(CAR_DIR));
	// 		std::vector<std::string> dir = DPhy::split(raw[i], '/');
	// 		for(int j = 0 ; j < dir.size(); j++) {
	// 			std::regex re(dir[j]);
	// 			std::vector<std::string> path_new;

	// 			for(int k = 0; k < path.size(); k++) {
	// 				std::vector<std::string> path_temp;

	// 				for(auto &p: std::experimental::filesystem::directory_iterator(path[k])) {
	// 					if(std::regex_match(p.path().filename().string(), re)) {
	// 						path_temp.push_back(p.path().filename().string());

	// 					}
	// 				}
	// 				std::sort(path_temp.begin(), path_temp.end(), compare_string_with_number);
	// 				for(int l = 0; l < path_temp.size(); l++) {
	// 					path_temp[l] = path[k] + std::string("/") + path_temp[l];
	// 					path_new.push_back(path_temp[l]);
	// 				}
	// 			}
	// 			path = path_new;
	// 		}
	// 		for(int j = 0; j < path.size(); j++) {
	// 			motion.push_back(path[j]);
	// 			std::cout << path[j] << std::endl;
	// 		}
	// 	}
	// 	if(type.compare("rec")==0) {
	// 		RecordWindow* rwindow = new RecordWindow(motion);
	// 		glutInit(&argc, argv);
	// 		rwindow->InitWindow(1600,900,"Render");
	// 		glutMainLoop();
	// 	} else {
	// 		SeqRecordWindow* rwindow = new SeqRecordWindow(motion);
	// 		glutInit(&argc, argv);
	// 		rwindow->InitWindow(1600,900,"Render");
	// 		glutMainLoop();
	// 	}
	// } if(type.compare("reg")==0) {
	// 	RegressionWindow* rgwindow = new RegressionWindow(ref, network);
	// 	glutInit(&argc, argv);
	// 	rgwindow->InitWindow(1600,900,"Render");
	// 	glutMainLoop();
	// }
	glutInit(&argc,argv);
	QApplication a(argc, argv);
    
    MainWindow* main_window = new MainWindow(ref, network);
    main_window->resize(2560,1440);
    main_window->show();
    return a.exec();
}
