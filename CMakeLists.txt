cmake_minimum_required(VERSION 3.1)
project(perceptron C)

set(CMAKE_C_STANDARD 99)

add_executable(perceptron standard.c)
add_executable(adaline adaline.c)
