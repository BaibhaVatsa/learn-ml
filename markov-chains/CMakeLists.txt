cmake_minimum_required(VERSION 3.14)
project(markov_chains)

set(CMAKE_CXX_STANDARD 17)

add_executable(markov_chains main.cpp Dictionary.cpp Dictionary.h)