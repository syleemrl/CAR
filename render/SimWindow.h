#ifndef __VMCON_SIM_WINDOW_H__
#define __VMCON_SIM_WINDOW_H__
#include "Camera.h"
#include "GLUTWindow.h"
#include "GLfunctions.h"
#include "Controller.h"
#include "Character.h"
#include "BVH.h"
#include "DART_interface.h"
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

class SimWindow : public GUI::GLUTWindow
{
public:
	/// Constructor.
	SimWindow(std::string motion, std::string network="");

	/// World object pointer
	dart::simulation::WorldPtr mWorld;
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

	/// Set the skeleton positions in mWorld to the positions at n frame.
	void SetFrame(int n);

	/// Set the skeleton positions in mWorld to the postions at the next frame.
	void NextFrame();

	/// Set the skeleton positions in mWorld to the postions at the previous frame.
	void PrevFrame();
	void MemoryClear();
	void Save();
	
	bool mIsRotate;
	bool mIsAuto;
	bool mTrackCamera;
	bool mDrawRef;
	bool mDrawOutput;
	bool mRunPPO;

	double mTimeStep;
	int mCurFrame;
	int mTotalFrame;
	double mReward;
		
	std::vector<Eigen::VectorXd> mMemory, mMemoryRef, mMemoryRefContact;
	Eigen::VectorXd mRefContact;

	DPhy::Controller* mController;
	DPhy::Character* mRef;
	DPhy::BVH* mBVH;

	p::object mPPO;
};

#endif