#include "../include/Utils.hpp"
#include <sstream>

std::vector<std::string> dirs = {"DOWN", "LEFT", "RIGHT", "UP"};

std::map<std::string, Coord> dir_vec = {
    {"RIGHT", {1, 0}},
    {"LEFT", {-1, 0}},
    {"UP", {0, -1}},
    {"DOWN", {0, 1}}};

std::string get_inverse(std::string dir)
{
    if (dir == "UP")
        return "DOWN";
    if (dir == "DOWN")
        return "UP";
    if (dir == "LEFT")
        return "RIGHT";
    return "LEFT";
}

std::vector<std::string> splitStringStream(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}