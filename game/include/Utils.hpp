#pragma once

#include <vector>
#include <string>
#include <map>
#include "Coord.hpp"

std::vector<std::string> splitStringStream(const std::string &s, char delimiter);
std::string get_inverse(std::string dir);

extern std::vector<std::string> dirs;
extern std::map<std::string, Coord> dir_vec;