#pragma once

struct Coord
{
    int x, y;
    Coord(int x = 0, int y = 0);

    bool operator<(const Coord &other) const;
    bool operator==(const Coord &other) const;
};