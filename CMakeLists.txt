cmake_minimum_required(VERSION 3.10)
project(SnakeAlphaZero)

set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(PYBIND11_INCLUDES)

add_library(snake_engine MODULE 
    back/src/bindings.cpp 
    back/src/Game.cpp 
    back/src/Grid.cpp 
    back/src/Coord.cpp 
    back/src/Utils.cpp
)

target_include_directories(snake_engine PRIVATE back/include)

target_compile_options(snake_engine PRIVATE ${PYBIND11_INCLUDES})
target_link_libraries(snake_engine PRIVATE Python::Module)

set_target_properties(snake_engine PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(snake_engine PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(snake_engine PROPERTIES SUFFIX ".so")
endif()