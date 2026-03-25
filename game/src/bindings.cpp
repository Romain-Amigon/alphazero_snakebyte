#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/Game.hpp"
#include "../include/Grid.hpp"
#include "../include/Coord.hpp"

namespace py = pybind11;

PYBIND11_MODULE(snake_engine, m)
{

    py::class_<Coord>(m, "Coord")
        .def(py::init<int, int>())
        .def_readwrite("x", &Coord::x)
        .def_readwrite("y", &Coord::y)
        .def("__lt__", &Coord::operator<)
        .def("__eq__", &Coord::operator==)
        .def("__hash__", [](const Coord &c)
             { return py::hash(py::make_tuple(c.x, c.y)); });

    py::enum_<Tile::Type>(m, "TileType")
        .value("TYPE_EMPTY", Tile::TYPE_EMPTY)
        .value("TYPE_WALL", Tile::TYPE_WALL)
        .export_values();

    py::class_<Tile>(m, "Tile")
        .def(py::init<>())
        .def("getType", &Tile::getType)
        .def("setType", &Tile::setType);

    py::class_<Grid>(m, "Grid")
        .def(py::init<int, int>())
        .def_readwrite("width", &Grid::width)
        .def_readwrite("height", &Grid::height)
        .def_readwrite("cells", &Grid::cells)
        .def_readwrite("apples", &Grid::apples)
        .def_readwrite("spawns", &Grid::spawns)
        .def("get", py::overload_cast<int, int>(&Grid::get, py::const_));

    py::class_<Bot>(m, "Bot")
        .def(py::init<>())
        .def_readwrite("id", &Bot::id)
        .def_readwrite("body", &Bot::body)
        .def_readwrite("last_move", &Bot::last_move);

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def_readwrite("width", &GameState::width)
        .def_readwrite("height", &GameState::height)
        .def_readwrite("grid", &GameState::grid)
        .def_readwrite("bots1", &GameState::bots1)
        .def_readwrite("bots2", &GameState::bots2)
        .def("copy", &GameState::copy)
        .def("step", &GameState::step);
}