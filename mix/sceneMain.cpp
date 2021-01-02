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
#include "MetaController.h"
#include "SceneMainWindow.h"

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

// bool compare_string_with_number(const std::string& s1, const std::string& s2) {
// 	int length = std::min(s1.length(), s2.length());
// 	for(int i = 0; i < length; i++) {
// 		if(s1.at(i) >= 48 && s1.at(i) <= 57 && s2.at(i) >= 48 && s2.at(i) <= 57) {
// 			int n1 = 0, n2 = 0;
// 			n1 = (int)s1.at(i) - 48;
// 			n2 = (int)s2.at(i) - 48;

// 			int l1 = 1, l2 = 1;
// 			while(s1.length() > i + l1 && s1.at(i + l1) >= 48 && s1.at(i + l1) <= 57 ) {
// 				n1 *= 10;
// 				n1 += (int)s1.at(i + l1) - 48;
// 				l1 += 1;
// 			}

// 			while(s2.length() > i + l2 &&  s2.at(i + l2) >= 48 && s2.at(i + l2) <= 57 ) {
// 				n2 *= 10;
// 				n2 += (int)s2.at(i + l2) - 48;
// 				l2 += 1;
// 			}

// 			if(n1 < n2) {
// 				return true;
// 			} else if(n1 > n2) {
// 				return false;
// 			}

// 			i += (l1 - 1);
// 		}
// 		else {
// 			if(s1.at(i) < s2.at(i)) {
// 				return true;
// 			} else if (s1.at(i) > s2.at(i)) {
// 				return true;
// 			}
// 		} 
// 	}
// 	return false;
// }
int main(int argc,char** argv)
{
    std::cout<<"Main!!"<<std::endl;
	boost::program_options::options_description desc("allowed options");
	desc.add_options()
	("reg,r",boost::program_options::value<std::string>())
	("bvh,b",boost::program_options::value<std::string>())
	("ppo,p",boost::program_options::value<std::string>())
	;

	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	std::string ppo="", bvh="", reg="";

	if(vm.count("ppo")) {
		ppo = vm["ppo"].as<std::string>();
	}
	if(vm.count("reg")) {
		reg = vm["reg"].as<std::string>();
	}
	if(vm.count("bvh")) {
		bvh = vm["bvh"].as<std::string>();
	}

	glutInit(&argc,argv);
	QApplication a(argc, argv);
    
    SceneMainWindow* main_window = new SceneMainWindow(bvh, ppo, reg);
    main_window->resize(2560,1440);
    main_window->show();
    return a.exec();

}
