#!/bin/bash


# Compile and run the OpenCL program
clang -framework OpenCL -o compiled_version.o $1
./compiled_version.o