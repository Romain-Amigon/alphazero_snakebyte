#include <iostream>
#include "../include/Game.hpp"
#include "../include/Grid.hpp"

int main()
{
    GridMaker maker(3);
    Grid g = maker.make();

    maker.printGridConsole();

    std::cout << "\nGeneration terminee !" << std::endl;
    return 0;
}