#include "Humanoid.h"

namespace DPhy
{
Humanoid::
Humanoid(const std::string& path)
	:Character(path)
{
	Character::LoadBVHMap(path);
}
}