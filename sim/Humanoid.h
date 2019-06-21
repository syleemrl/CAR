#ifndef __DEEP_PHYSICS_HUMANOID_H__
#define __DEEP_PHYSICS_HUMANOID_H__
#include "Character.h"
namespace DPhy
{

class Humanoid : public Character
{
public:
	Humanoid(const std::string& path = std::string(DPHY_DIR)+std::string("/character/humanoid.xml"));
};
}
#endif