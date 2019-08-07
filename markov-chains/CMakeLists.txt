cmake_minimum_required(VERSION 3.14)
project(markov_chains)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Werror -Wextra -pedantic -pedantic-errors -lstdc++fs")

add_executable(markov_chains main.cpp Dictionary.cpp Dictionary.h)