#!/bin/bash

# Determine OS type
OS=$(uname)

# Set the output file name
OUTPUT="compiled_version.o"
INPUT=$1;

if [ "$OS" == "Darwin" ]; then
    echo "Compiling for macOS..."
    clang -o $OUTPUT -framework OpenCL $INPUT;
elif [ "$OS" == "Linux" ]; then
    echo "Compiling for Linux..."
    gcc -o $OUTPUT -lOpenCL $INPUT;
else
    echo "Unsupported OS. Please use macOS, Linux."
    exit 1
fi

# Check if the compilation succeeded
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    ./$OUTPUT
else
    echo "Compilation failed."
    exit 1
fi





# clang -framework OpenCL -o compiled_version.o $1
# ./compiled_version.o