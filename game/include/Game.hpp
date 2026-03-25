#pragma once

#include <vector>
#include <string>
#include <map>
#include <set>
#include "Coord.hpp"
#include "../include/Grid.hpp"
struct Bot
{
    int id;
    std::vector<Coord> body;
    std::string last_move = "UP";
    int compt_same_move = 0;
    Coord tail_reserve = {-1, -1};

    void set_body(std::string body_str);
};

struct GameState
{
    int width, height;
    Grid grid;
    std::map<int, Bot> bots1;
    std::map<int, Bot> bots2;

    GameState();
    GameState copy() const;
    void step(const std::map<int, std::string> &my_actions, const std::map<int, std::string> &opp_actions = {});
};

class Game
{
public:
    GameState state;
    int nmb_bot_total;
    std::vector<std::vector<std::string>> grid;
    int width;
    int height;

    Game(int w, int h);
};

bool is_supported(const Bot &bot, const std::set<Coord> &solids);