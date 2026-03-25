#pragma once

#include <vector>
#include <set>
#include <map>
#include <random>
#include "Coord.hpp"

struct Tile
{
    enum Type
    {
        TYPE_EMPTY,
        TYPE_WALL
    };
    Type type = TYPE_EMPTY;

    void setType(Type t);
    Type getType() const;
};

struct Grid
{
    int width, height;
    std::map<Coord, Tile> cells;
    std::set<Coord> apples;
    std::set<Coord> spawns;

    enum Adjacency
    {
        ADJACENCY_4,
        ADJACENCY_8
    };

    Grid(int w = 0, int h = 0);

    Tile &get(int x, int y);
    Tile &get(Coord c);

    const Tile &get(int x, int y) const;
    const Tile &get(Coord c) const;

    bool isValid(Coord c) const;
    Coord opposite(Coord c);

    std::vector<std::set<Coord>> detectAirPockets();
    std::vector<Coord> getNeighbours(Coord c, Adjacency adj = ADJACENCY_4);
    std::vector<Coord> detectLowestIsland();
};

class GridMaker
{
    int leagueLevel;
    Grid grid;
    std::mt19937 rng;

    const int MIN_GRID_HEIGHT = 10;
    const int MAX_GRID_HEIGHT = 24;
    const float ASPECT_RATIO = 1.8f;
    const int SPAWN_HEIGHT = 3;
    const int DESIRED_SPAWNS = 4;

    std::vector<Coord> getFreeAbove(Coord c, int by);
    void checkGrid();

public:
    GridMaker(int league);
    Grid make();
    void printGridConsole();
};