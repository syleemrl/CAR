#ifndef __DEEP_PHYSICS_GROUND_H__
#define __DEEP_PHYSICS_GROUND_H__
#include "Character.h"
namespace DPhy
{
class Ground : public Character
{
public:
	Ground(const std::string& path = std::string(DPHY_DIR)+std::string("/character/ground.xml"));
};

}

#endif