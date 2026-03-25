#include "Grid.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>    // std::to_string
#include <stdexcept> // std::runtime_error

// ─────────────────────────────────────────
//  Tile
// ─────────────────────────────────────────

void Tile::setType(Type t) { type = t; }
Tile::Type Tile::getType() const { return type; }

// ─────────────────────────────────────────
//  Grid
// ─────────────────────────────────────────

Grid::Grid(int w, int h) : width(w), height(h)
{
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            cells[Coord(x, y)] = Tile{};
}

Tile &Grid::get(int x, int y)
{
    return cells[Coord(x, y)];
}

Tile &Grid::get(Coord c)
{
    return cells[c];
}

const Tile &Grid::get(int x, int y) const
{
    return cells.at(Coord(x, y));
}

const Tile &Grid::get(Coord c) const
{
    return cells.at(c);
}

bool Grid::isValid(Coord c) const
{
    return c.x >= 0 && c.x < width && c.y >= 0 && c.y < height;
}

Coord Grid::opposite(Coord c)
{
    return Coord(width - 1 - c.x, c.y);
}

std::vector<Coord> Grid::getNeighbours(Coord c, Adjacency adj)
{
    std::vector<Coord> res;
    int dx[] = {0, 1, 0, -1, -1, 1, 1, -1};
    int dy[] = {-1, 0, 1, 0, -1, -1, 1, 1};
    int limit = (adj == ADJACENCY_4) ? 4 : 8;
    for (int i = 0; i < limit; ++i)
    {
        Coord n(c.x + dx[i], c.y + dy[i]);
        if (isValid(n))
            res.push_back(n);
    }
    return res;
}

std::vector<std::set<Coord>> Grid::detectAirPockets()
{
    std::vector<std::set<Coord>> pockets;
    std::set<Coord> visited;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            Coord start(x, y);
            if (get(start).getType() != Tile::TYPE_EMPTY)
                continue;
            if (visited.count(start))
                continue;

            std::set<Coord> pocket;
            std::vector<Coord> q;
            q.push_back(start);
            visited.insert(start);
            pocket.insert(start);

            size_t head = 0;
            while (head < q.size())
            {
                Coord curr = q[head++];
                for (Coord n : getNeighbours(curr, ADJACENCY_4))
                {
                    if (get(n).getType() == Tile::TYPE_EMPTY && !visited.count(n))
                    {
                        visited.insert(n);
                        pocket.insert(n);
                        q.push_back(n);
                    }
                }
            }
            pockets.push_back(pocket);
        }
    }
    return pockets;
}

std::vector<Coord> Grid::detectLowestIsland()
{
    // CORRECTION : On cherche les îles de MURS, pas les poches d'air !
    std::vector<std::set<Coord>> islands;
    std::set<Coord> visited;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            Coord start(x, y);
            if (get(start).getType() != Tile::TYPE_WALL)
                continue;
            if (visited.count(start))
                continue;

            std::set<Coord> island;
            std::vector<Coord> q;
            q.push_back(start);
            visited.insert(start);
            island.insert(start);

            size_t head = 0;
            while (head < q.size())
            {
                Coord curr = q[head++];
                for (Coord n : getNeighbours(curr, ADJACENCY_4))
                {
                    if (get(n).getType() == Tile::TYPE_WALL && !visited.count(n))
                    {
                        visited.insert(n);
                        island.insert(n);
                        q.push_back(n);
                    }
                }
            }
            islands.push_back(island);
        }
    }

    if (islands.empty())
        return {};

    std::vector<Coord> lowest;
    int maxY = -1;
    for (const auto &island : islands)
    {
        int currentMaxY = -1;
        for (const Coord &c : island)
            if (c.y > currentMaxY)
                currentMaxY = c.y;

        if (currentMaxY > maxY)
        {
            maxY = currentMaxY;
            lowest = std::vector<Coord>(island.begin(), island.end());
        }
    }
    return lowest;
}

// ─────────────────────────────────────────
//  GridMaker – helpers
// ─────────────────────────────────────────

GridMaker::GridMaker(int league) : leagueLevel(league)
{
    std::random_device rd;
    rng = std::mt19937(rd());
}

std::vector<Coord> GridMaker::getFreeAbove(Coord c, int by)
{
    std::vector<Coord> res;
    for (int i = 1; i <= by; ++i)
    {
        Coord above(c.x, c.y - i);
        if (grid.isValid(above) && grid.get(above).getType() == Tile::TYPE_EMPTY)
            res.push_back(above);
        else
            break;
    }
    return res;
}

void GridMaker::checkGrid()
{
    for (const Coord &c : grid.apples)
        if (grid.get(c).getType() != Tile::TYPE_EMPTY)
            throw std::runtime_error("Apple on wall at " + std::to_string(c.x) + "," + std::to_string(c.y));

    std::set<Coord> unique(grid.apples.begin(), grid.apples.end());
    if (unique.size() != grid.apples.size())
        throw std::runtime_error("Duplicate apples");
}

// ─────────────────────────────────────────
//  GridMaker::make
// ─────────────────────────────────────────

Grid GridMaker::make()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double skew;
    if (leagueLevel == 1)
        skew = 2.0;
    else if (leagueLevel == 2)
        skew = 1.0;
    else if (leagueLevel == 3)
        skew = 0.8;
    else
        skew = 0.3;

    int height = MIN_GRID_HEIGHT + (int)std::round(std::pow(dist(rng), skew) * (MAX_GRID_HEIGHT - MIN_GRID_HEIGHT));
    int width = (int)std::round(height * ASPECT_RATIO);
    if (width % 2 != 0)
        width += 1;

    grid = Grid(width, height);

    for (int x = 0; x < width; ++x)
        grid.get(x, height - 1).setType(Tile::TYPE_WALL);

    double b = 5.0 + dist(rng) * 10.0;
    for (int y = height - 2; y >= 0; --y)
    {
        double yNorm = static_cast<double>(height - 1 - y) / (height - 1);
        double blockChance = 1.0 / (yNorm + 0.1) / b;
        for (int x = 0; x < width; ++x)
            if (dist(rng) < blockChance)
                grid.get(x, y).setType(Tile::TYPE_WALL);
    }

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            Coord c(x, y);
            grid.get(grid.opposite(c)).setType(grid.get(c).getType());
        }

    {
        auto islands = grid.detectAirPockets();
        for (const auto &island : islands)
            if (island.size() < 10)
                for (const Coord &c : island)
                    grid.get(c).setType(Tile::TYPE_WALL);
    }

    bool somethingDestroyed = true;
    while (somethingDestroyed)
    {
        somethingDestroyed = false;
        std::vector<Coord> keys;
        keys.reserve(grid.cells.size());
        for (const auto &pair : grid.cells)
            keys.push_back(pair.first);

        for (const Coord &c : keys)
        {
            if (grid.get(c).getType() == Tile::TYPE_WALL)
                continue;

            std::vector<Coord> neighbourWalls;
            for (const Coord &n : grid.getNeighbours(c))
                if (grid.get(n).getType() == Tile::TYPE_WALL)
                    neighbourWalls.push_back(n);

            if (neighbourWalls.size() >= 3)
            {
                std::vector<Coord> destroyable;
                for (const Coord &n : neighbourWalls)
                    if (n.y <= c.y)
                        destroyable.push_back(n);

                if (!destroyable.empty())
                {
                    std::shuffle(destroyable.begin(), destroyable.end(), rng);
                    grid.get(destroyable[0]).setType(Tile::TYPE_EMPTY);
                    grid.get(grid.opposite(destroyable[0])).setType(Tile::TYPE_EMPTY);
                    somethingDestroyed = true;
                }
            }
        }
    }

    {
        std::vector<Coord> lowestIsland = grid.detectLowestIsland();
        int lowerBy = 0;
        bool canLower = true;
        while (canLower)
        {
            for (int x = 0; x < width; ++x)
            {
                Coord c(x, height - 1 - (lowerBy + 1));
                if (!std::count(lowestIsland.begin(), lowestIsland.end(), c))
                {
                    canLower = false;
                    break;
                }
            }
            if (canLower)
                lowerBy++;
        }

        if (lowerBy >= 2)
        {
            std::uniform_int_distribution<int> intDist(2, lowerBy);
            lowerBy = intDist(rng);
        }

        for (const Coord &c : lowestIsland)
        {
            grid.get(c).setType(Tile::TYPE_EMPTY);
            grid.get(grid.opposite(c)).setType(Tile::TYPE_EMPTY);
        }
        for (const Coord &c : lowestIsland)
        {
            Coord lowered(c.x, c.y + lowerBy);
            if (grid.isValid(lowered))
            {
                grid.get(lowered).setType(Tile::TYPE_WALL);
                grid.get(grid.opposite(lowered)).setType(Tile::TYPE_WALL);
            }
        }
    }

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width / 2; ++x)
        {
            Coord c(x, y);
            if (grid.get(c).getType() == Tile::TYPE_EMPTY && dist(rng) < 0.025)
            {
                grid.apples.insert(c);
                grid.apples.insert(grid.opposite(c));
            }
        }

    if (grid.apples.size() < 8)
    {
        grid.apples.clear();
        std::vector<Coord> freeTiles;
        for (const auto &pair : grid.cells)
            if (pair.second.getType() == Tile::TYPE_EMPTY)
                freeTiles.push_back(pair.first);

        std::shuffle(freeTiles.begin(), freeTiles.end(), rng);
        int minApples = std::max(4, (int)(0.025 * freeTiles.size()));

        while ((int)grid.apples.size() < minApples * 2 && !freeTiles.empty())
        {
            Coord c = freeTiles.back();
            freeTiles.pop_back();
            Coord opp = grid.opposite(c);
            grid.apples.insert(c);
            grid.apples.insert(opp);
            freeTiles.erase(std::remove(freeTiles.begin(), freeTiles.end(), opp), freeTiles.end());
        }
    }

    {
        std::vector<Coord> keys;
        for (const auto &pair : grid.cells)
            keys.push_back(pair.first);

        for (const Coord &c : keys)
        {
            if (grid.get(c).getType() == Tile::TYPE_EMPTY)
                continue;

            int wallNeighbours = 0;
            for (const Coord &n : grid.getNeighbours(c, Grid::ADJACENCY_8))
                if (grid.get(n).getType() == Tile::TYPE_WALL)
                    wallNeighbours++;

            if (wallNeighbours == 0)
            {
                grid.get(c).setType(Tile::TYPE_EMPTY);
                grid.get(grid.opposite(c)).setType(Tile::TYPE_EMPTY);
                grid.apples.insert(c);
                grid.apples.insert(grid.opposite(c));
            }
        }
    }

    {
        std::vector<Coord> potentialSpawns;
        for (const auto &pair : grid.cells)
        {
            Coord c = pair.first;
            if (grid.get(c).getType() == Tile::TYPE_WALL && (int)getFreeAbove(c, SPAWN_HEIGHT).size() >= SPAWN_HEIGHT)
                potentialSpawns.push_back(c);
        }
        std::shuffle(potentialSpawns.begin(), potentialSpawns.end(), rng);

        int desiredSpawns = DESIRED_SPAWNS;
        if (height <= 15)
            desiredSpawns--;
        if (height <= 10)
            desiredSpawns--;

        for (auto it = potentialSpawns.begin(); desiredSpawns > 0 && it != potentialSpawns.end(); ++it)
        {
            Coord spawn = *it;
            std::vector<Coord> spawnLoc = getFreeAbove(spawn, SPAWN_HEIGHT);
            bool tooClose = false;

            for (const Coord &c : spawnLoc)
            {
                if (c.x == width / 2 - 1 || c.x == width / 2)
                {
                    tooClose = true;
                    break;
                }
                for (const Coord &n : grid.getNeighbours(c, Grid::ADJACENCY_8))
                {
                    if (grid.spawns.count(n) || grid.spawns.count(grid.opposite(n)))
                    {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose)
                    break;
            }
            if (tooClose)
                continue;

            for (const Coord &ac : spawnLoc)
            {
                grid.spawns.insert(ac);
                grid.apples.erase(ac);
                grid.apples.erase(grid.opposite(ac));
            }
            desiredSpawns--;
        }
    }

    checkGrid();
    return grid;
}

// ─────────────────────────────────────────
//  Debug
// ─────────────────────────────────────────

void GridMaker::printGridConsole()
{
    for (int y = 0; y < grid.height; ++y)
    {
        for (int x = 0; x < grid.width; ++x)
        {
            Coord c(x, y);
            if (grid.spawns.count(c))
                std::cout << '1';
            else if (grid.spawns.count(grid.opposite(c)))
                std::cout << '2';
            else if (grid.apples.count(c))
                std::cout << 'A';
            else if (grid.get(c).getType() == Tile::TYPE_WALL)
                std::cout << '#';
            else
                std::cout << '.';
        }
        std::cout << '\n';
    }
}