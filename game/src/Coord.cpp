#include "../include/Coord.hpp"

Coord::Coord(int x, int y) : x(x), y(y) {}

bool Coord::operator<(const Coord &other) const
{
    if (x != other.x)
        return x < other.x;
    return y < other.y;
}

bool Coord::operator==(const Coord &other) const
{
    return x == other.x && y == other.y;
}