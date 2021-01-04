#include "Camera.h"
#include <GL/glut.h>
#include <iostream>
Camera::
Camera(int w, int h)
	:fovy(60.0),lookAt(Eigen::Vector3d(0,1.5,0)),eye(Eigen::Vector3d(0,1.0, 3)),up(Eigen::Vector3d(0,1,0)),mw(w),mh(h)
{

}
void
Camera::
SetSize(int w,int h)
{
	mw = w;
	mh = h;
}
void
Camera::
SetCamera(const Eigen::Vector3d& lookAt,const Eigen::Vector3d& eye,const Eigen::Vector3d& up)
{
	this->lookAt = lookAt, this->eye = eye, this->up = up;
}
void
Camera::
Apply()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, (GLfloat)mw / (GLfloat)mh, 0.01, 1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye.x(), eye.y(), eye.z(),
		lookAt.x(), lookAt.y(), lookAt.z(),
		up.x(), up.y(), up.z());
}

void
Camera::
Pan(int x,int y,int prev_x,int prev_y)
{
	double delta = ((double)prev_y - (double)y)/15.0;
	Eigen::Vector3d vec = (lookAt - eye);
	double scale = vec.norm();
	scale = std::max((scale - delta),1.0);
	Eigen::Vector3d vd = (scale - delta) * (lookAt - eye).normalized();	
	// eye = eye + vd;
	// lookAt = lookAt;// + vd;
	eye = lookAt - vd;
}
void
Camera::
Zoom(int x,int y,int prev_x,int prev_y)
{
	double delta = (double)prev_y - (double)y;
	fovy += delta/20.0;
}
void
Camera::
Rotate(int x,int y,int prev_x,int prev_y)
{
	GLint w = glutGet(GLUT_WINDOW_WIDTH);
	GLint h = glutGet(GLUT_WINDOW_HEIGHT);

	// Eigen::Vector3d prevPoint = GetTrackballPoint(prev_x,prev_y,w,h);
	// Eigen::Vector3d curPoint = GetTrackballPoint(x,y,w,h);
	// Eigen::Vector3d rotVec = curPoint.cross(prevPoint);

	// rotVec = UnProject(rotVec);
	// double cosT = curPoint.dot(prevPoint) / (curPoint.norm()*prevPoint.norm());
	// double sinT = (curPoint.cross(prevPoint)).norm() / (curPoint.norm()*prevPoint.norm());

	// double angle = -atan2(sinT, cosT);

	double rad = std::min(mw, mh) / 2.0;
	double dx = (double)x - (double)prev_x;
	double dy = (double)y - (double)prev_y;
	double angleY = atan2(dx * 0.5, rad);
	double angleX = atan2(dy * 0.5, rad);

	Eigen::Vector3d n = this->lookAt - this->eye;
	Eigen::Vector3d axisX = Eigen::Vector3d::UnitY().cross(n.normalized());
	n = Eigen::Quaterniond(Eigen::AngleAxisd(-angleY, Eigen::Vector3d::UnitY()))._transformVector(n);
	n = Eigen::Quaterniond(Eigen::AngleAxisd(angleX, axisX))._transformVector(n);
	// n = Rotateq(n, Eigen::Vector3d::UnitY(), angleY);
	// n = Rotateq(n, Eigen::Vector3d::UnitX(), angleX);
	// this->up = Rotateq(this->up, rotVec, angle);
	this->eye = this->lookAt - n;
}
void 
Camera::
SetCenter(Eigen::Vector3d c){
	Eigen::Vector3d delta = c - lookAt;
	lookAt += delta; eye += delta;
}
void
Camera::
Translate(int x,int y,int prev_x,int prev_y)
{
	Eigen::Vector3d delta((double)x - (double)prev_x, (double)y - (double)prev_y, 0);
	// delta = UnProject(delta) / 200.0;

	Eigen::Vector3d yvec = (lookAt - eye);
	yvec[1] = 0;
	double scale = yvec.norm()/1000.;
	yvec.normalize();
	Eigen::Vector3d xvec = -yvec.cross(up);
	xvec.normalize();

	delta = delta[0]*xvec*scale + delta[1]*yvec*scale;

	lookAt += delta; eye += delta;
}
void
Camera::
SetLookAt(const Eigen::Vector3d& lookAt)
{
	this->lookAt = lookAt;
	this->eye = lookAt;
	eye[2] += 2;
}
#include <iostream>
Eigen::Vector3d
Camera::
GetDeltaPosition(int x,int y,int prev_x,int prev_y)
{
	Eigen::Vector3d delta((double)x - (double)prev_x, (double)y - (double)prev_y, 0);
	delta = UnProject(delta) / 200.0;
	return delta;
}
Eigen::Vector3d
Camera::
Rotateq(const Eigen::Vector3d& target, const Eigen::Vector3d& rotateVector,double angle)
{
	Eigen::Vector3d rv = rotateVector.normalized();

	Eigen::Quaternion<double> rot(cos(angle / 2.0), sin(angle / 2.0)*rv.x(), sin(angle / 2.0)*rv.y(), sin(angle / 2.0)*rv.z());
	rot.normalize();
	Eigen::Quaternion<double> tar(0, target.x(), target.y(), target.z());


	tar = rot.inverse()*tar*rot;

	return Eigen::Vector3d(tar.x(), tar.y(), tar.z());
}
Eigen::Vector3d
Camera::
GetTrackballPoint(int mouseX, int mouseY,int w,int h)
{
	double rad = sqrt((double)(w*w+h*h)) / 2.0;
	double dx = (double)(mouseX)-(double)w / 2.0;
	double dy = (double)(mouseY)-(double)h / 2.0;
	double dx2pdy2 = dx*dx + dy*dy;

	if (rad*rad - dx2pdy2 <= 0)
		return Eigen::Vector3d(dx/rad, dy/rad, 0);
	else
		return Eigen::Vector3d(dx/rad, dy/rad, sqrt(rad*rad - dx*dx - dy*dy)/rad);
}
Eigen::Vector3d
Camera::
UnProject(const Eigen::Vector3d& vec)
{
	Eigen::Vector3d n = lookAt - eye;
	n.normalize();
	
	Eigen::Vector3d v = up.cross(n);
	v.normalize();

	Eigen::Vector3d u = n.cross(v);
	u.normalize();

	return vec.z()*n + vec.x()*v + vec.y()*u;
}