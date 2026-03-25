#include "../include/Game.hpp"
#include "../include/Utils.hpp"
#include <algorithm>

void Bot::set_body(std::string body_str)
{
    body.clear();
    std::vector<std::string> Coords = splitStringStream(body_str, ':');
    for (const std::string &c : Coords)
    {
        std::vector<std::string> couple = splitStringStream(c, ',');
        if (couple.size() == 2)
        {
            Coord p;
            p.x = std::stoi(couple[0]);
            p.y = std::stoi(couple[1]);
            body.push_back(p);
        }
    }
}

GameState::GameState()
{
    GridMaker maker(3);
    grid = maker.make();
    width = grid.width;
    height = grid.height;

    std::map<int, std::vector<Coord>> pillars;
    for (Coord c : grid.spawns)
    {
        pillars[c.x].push_back(c);
    }

    int bot_id = 0;
    for (auto &kv : pillars)
    {
        std::vector<Coord> &parts = kv.second;

        std::sort(parts.begin(), parts.end(), [](const Coord &a, const Coord &b)
                  { return a.y < b.y; });

        Bot b1, b2;
        b1.id = bot_id;
        b2.id = bot_id + 1;
        b1.last_move = "UP";
        b2.last_move = "UP";

        for (Coord c : parts)
        {
            b1.body.push_back(c);
            b2.body.push_back(grid.opposite(c));
        }

        bots1[b1.id] = b1;
        bots2[b2.id] = b2;
        bot_id += 2;
    }
}

GameState GameState::copy() const
{
    return *this;
}

bool is_supported(const Bot &bot, const Grid &grid, const std::set<Coord> &dynamic_solids)
{
    for (const Coord &part : bot.body)
    {
        Coord under = {part.x, part.y + 1};

        if (grid.get(under).getType() == Tile::TYPE_WALL || grid.apples.count(under) || dynamic_solids.count(under))
        {
            return true;
        }
    }
    return false;
}

void GameState::step(const std::map<int, std::string> &my_actions, const std::map<int, std::string> &opp_actions)
{
    std::map<int, std::string> action_dict;
    for (const auto &kv : my_actions)
        action_dict[kv.first] = kv.second;
    for (const auto &kv : opp_actions)
        action_dict[kv.first] = kv.second;

    auto apply_moves = [&](std::map<int, Bot> &bots_dict)
    {
        for (auto &kv : bots_dict)
        {
            int bot_id = kv.first;
            Bot &bot = kv.second;
            std::string d = bot.last_move;
            if (action_dict.count(bot_id))
                d = action_dict[bot_id];
            bot.last_move = d;

            Coord dir = dir_vec[d];
            bot.tail_reserve = bot.body.back();

            Coord new_head = {bot.body[0].x + dir.x, bot.body[0].y + dir.y};

            for (size_t i = bot.body.size() - 1; i > 0; --i)
            {
                bot.body[i] = bot.body[i - 1];
            }
            bot.body[0] = new_head;
        }
    };

    apply_moves(bots1);
    apply_moves(bots2);

    std::set<Coord> all_body_parts;
    auto collect_parts = [&](const std::map<int, Bot> &bots_dict)
    {
        for (const auto &kv : bots_dict)
        {
            for (size_t i = 1; i < kv.second.body.size(); ++i)
            {
                all_body_parts.insert(kv.second.body[i]);
            }
        }
    };

    collect_parts(bots1);
    collect_parts(bots2);

    std::set<int> destroyed_heads;
    auto check_heads = [&](const std::map<int, Bot> &bots_dict)
    {
        for (const auto &kv : bots_dict)
        {
            Coord head = kv.second.body[0];
            if (grid.get(head).getType() == Tile::TYPE_WALL || all_body_parts.count(head))
            {
                destroyed_heads.insert(kv.first);
            }
        }
    };

    check_heads(bots1);
    check_heads(bots2);

    std::set<Coord> eaten_energy;
    auto eat = [&](std::map<int, Bot> &bots_dict)
    {
        for (auto &kv : bots_dict)
        {
            Coord head = kv.second.body[0];
            if (grid.apples.count(head) && !destroyed_heads.count(kv.first))
            {
                eaten_energy.insert(head);
                kv.second.body.push_back(kv.second.tail_reserve);
            }
        }
    };

    eat(bots1);
    eat(bots2);

    std::vector<int> my_to_delete, opp_to_delete;
    auto handle_destroyed = [&](std::map<int, Bot> &bots_dict, std::vector<int> &to_delete)
    {
        for (auto &kv : bots_dict)
        {
            if (destroyed_heads.count(kv.first))
            {
                kv.second.body.erase(kv.second.body.begin());
                if (kv.second.body.size() < 3)
                    to_delete.push_back(kv.first);
            }
        }
    };

    handle_destroyed(bots1, my_to_delete);
    handle_destroyed(bots2, opp_to_delete);

    for (int id : my_to_delete)
        bots1.erase(id);
    for (int id : opp_to_delete)
        bots2.erase(id);

    for (const Coord &e : eaten_energy)
        grid.apples.erase(e);

    bool falling = true;
    while (falling)
    {
        falling = false;
        std::set<Coord> dynamic_solids;

        auto apply_gravity = [&](std::map<int, Bot> &bots_dict)
        {
            for (auto &kv : bots_dict)
            {
                dynamic_solids.clear();

                for (const auto &bot : bots1)
                {
                    if (kv.first == bot.first)
                        continue;
                    for (const Coord &part : bot.second.body)
                        dynamic_solids.insert(part);
                }
                for (const auto &bot : bots2)
                {
                    if (kv.first == bot.first)
                        continue;
                    for (const Coord &part : bot.second.body)
                        dynamic_solids.insert(part);
                }

                if (!is_supported(kv.second, grid, dynamic_solids))
                {
                    for (Coord &part : kv.second.body)
                        part.y++;
                    falling = true;
                }
            }
        };

        apply_gravity(bots1);
        apply_gravity(bots2);
    }

    my_to_delete.clear();
    opp_to_delete.clear();

    auto check_bounds = [&](std::map<int, Bot> &bots_dict, std::vector<int> &to_delete)
    {
        for (const auto &kv : bots_dict)
        {
            bool in_bounds = false;
            for (const Coord &part : kv.second.body)
            {
                if (part.x >= 0 && part.x < width && part.y >= 0 && part.y < height)
                {
                    in_bounds = true;
                    break;
                }
            }
            if (!in_bounds)
                to_delete.push_back(kv.first);
        }
    };

    check_bounds(bots1, my_to_delete);
    check_bounds(bots2, opp_to_delete);

    for (int id : my_to_delete)
        bots1.erase(id);
    for (int id : opp_to_delete)
        bots2.erase(id);
}

Game::Game(int w, int h) : state(), width(w), height(h), nmb_bot_total(0)
{
    width = state.width;
    height = state.height;
}