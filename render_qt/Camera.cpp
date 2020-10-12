#include "Camera.h"
#include <GL/glut.h>
Camera::
Camera(int w,int h)
	:fovy(60.0),lookAt(Eigen::Vector3d(0,1,0)),eye(Eigen::Vector3d(4,2,2)),up(Eigen::Vector3d(0,1,0)),mw(w),mh(h)
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
	double delta = (double)prev_y - (double)y;
	delta = 1 - delta / 200.0;
	eye = lookAt - (lookAt - eye)*delta;
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
	Eigen::Vector3d prevPoint = GetTrackballPoint(prev_x,prev_y,mw,mh);
	Eigen::Vector3d curPoint = GetTrackballPoint(x,y,mw,mh);
	Eigen::Vector3d rotVec = curPoint.cross(prevPoint);

	if(rotVec.norm()<1E-6)
		return;
	rotVec = UnProject(rotVec);
	double cosT = curPoint.dot(prevPoint) / (curPoint.norm()*prevPoint.norm());
	double sinT = (curPoint.cross(prevPoint)).norm() / (curPoint.norm()*prevPoint.norm());

	double angle = -atan2(sinT, cosT);

	Eigen::Vector3d n = this->lookAt - this->eye;
	// if(rotVec[1]<0.0)
	// 	rotVec = -Eigen::Vector3d::UnitY();
	// else
	// 	rotVec = Eigen::Vector3d::UnitY();
	n = Rotateq(n, rotVec, angle);
	this->up = Rotateq(this->up, rotVec, angle);
	this->eye = this->lookAt - n;
}
void
Camera::
Translate(int x,int y,int prev_x,int prev_y)
{
	Eigen::Vector3d delta((double)x - (double)prev_x, (double)y - (double)prev_y, 0);
	delta = UnProject(delta) / 200.0;
	lookAt += delta; eye += delta;
}
void
Camera::
SetLookAt(const Eigen::Vector3d& lookAt)
{
	Eigen::Vector3d diff = lookAt - this->lookAt;
	this->lookAt =lookAt;
	this->eye = this->eye + diff;	
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