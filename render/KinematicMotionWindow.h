#ifndef __VMCON_KIN_WINDOW_H__
#define __VMCON_KIN_WINDOW_H__
#include "Camera.h"
#include "GLUTWindow.h"
#include "GLfunctions.h"
#include "Controller.h"
#include "Character.h"
#include "BVH.h"
#include "DART_interface.h"
#include "ReferenceManager.h"
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
/**
*
* @brief Modified SimWindow class in dart.
* @details Renders on window with the recorded data generated in sim.
*
*/
namespace p = boost::python;
namespace np = boost::python::numpy;

class KinematicMotionWindow : public GUI::GLUTWindow
{
public:
	/// Constructor.
	KinematicMotionWindow(std::vector<std::string> motion, std::vector<std::string> mode);

protected:
	/// Draw all the skeletons in mWorld. Lights and Camera are operated here.
	void DrawSkeletons();
	void DrawGround();
	void Display() override;

	/// The user interactions with keyboard.
	/// [ : Frame --
	/// ] : Frame ++
	/// r : Frame = 0
	/// C : Capture
	/// SPACE : Play
	/// ESC : exit
	void Keyboard(unsigned char key,int x,int y) override;

	/// Stores the data for SimWindow::Motion.
	void Mouse(int button, int state, int x, int y) override;

	/// The user interactions with mouse. Camera view is set here.
	void Motion(int x, int y) override;

	/// Reaction to window resizing.
	void Reshape(int w, int h) override;

	void Timer(int value) override;

	void Step();
	
	void Reset();
	/// Set the skeleton positions in mWorld to the positions at n frame.
	void SetFrame(int n);

	/// Set the skeleton positions in mWorld to the postions at the next frame.
	void NextFrame();

	/// Set the skeleton positions in mWorld to the postions at the previous frame.
	void PrevFrame();
	void MemoryClear();
	void Save(int n);
	
	bool mIsRotate;
	bool mIsAuto;
	bool mTrackCamera;
	std::vector<bool> mDrawRef;

	double mTimeStep;
	int mCurFrame;
	int mTotalFrame;

	std::vector<std::vector<Eigen::VectorXd>> mMemoryRef;
	std::vector<std::string> mode;
	std::vector<DPhy::Character*> mRef;
	std::vector<DPhy::ReferenceManager*> mReferenceManager;

	p::object mPPO;
};

#endif