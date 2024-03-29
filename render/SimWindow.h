#ifndef __VMCON_SIM_WINDOW_H__
#define __VMCON_SIM_WINDOW_H__
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

class SimWindow : public GUI::GLUTWindow
{
public:
	/// Constructor.
	SimWindow(std::string motion, std::string network="", std::string filename="");

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
	
	void Reset();
	/// Set the skeleton positions in mWorld to the positions at n frame.
	void SetFrame(int n);

	/// Set the skeleton positions in mWorld to the postions at the next frame.
	void NextFrame();

	/// Set the skeleton positions in mWorld to the postions at the previous frame.
	void PrevFrame();
	void MemoryClear();
	void Save(int n);
	
	void SaveReferenceData(std::string path);

	bool mIsRotate;
	bool mIsAuto;
	bool mTrackCamera;
	bool mDrawRef, mDrawRef2, mDrawRef3;
	bool mDrawOutput;
	bool mRunPPO;
	bool mWrap;
	
	double mTimeStep;
	int mCurFrame;
	int mTotalFrame;
	std::vector<double> mReward;
	double mSkelLength;
	double mRewardTotal;
	std::string mode;
	std::vector<Eigen::VectorXd> mMemory, mMemoryRef, mMemoryRef2, mMemoryRef3, mMemoryObj;
	std::vector<Eigen::Vector3d> mMemoryCOM, mMemoryCOMRef, mMemoryCOMRef2;
	std::vector<std::vector<Eigen::VectorXd>> mMemoryGRF;
	std::vector<std::pair<bool, bool>> mMemoryFootContact;
	std::pair<bool, bool> mFootContact;
	std::string filename;
	DPhy::Controller* mController;
	DPhy::Character* mCharacter;
	DPhy::Character* mRef;
	DPhy::Character* mRef2;
	DPhy::Character* mRef3;
	DPhy::Character* mObject;
	DPhy::ReferenceManager* mReferenceManager;
	DPhy::BVH* mBVH;

	p::object mPPO;
	p::object mRegression;

	int mPhaseCounter;
	double mPrevFrame;
};

#endif