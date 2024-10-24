# Use the Alpine slim base image
FROM alpine:3.20.3

# Install the required build tools, OpenCL, OpenCV, and related libraries
RUN apk update && apk add --no-cache \
    build-base \
    cmake \
    git \
    wget \
    opencl-icd-loader \
    opencl-dev \
    libstdc++ \
    opencv-dev \
    pkgconf

# Set the working directory
WORKDIR /app

# Copy the source code to the container
COPY main.cpp .
COPY kernal_code.cl .

# Copy include directory
COPY include include

# Build the C++ program with OpenCL and OpenCV (replace main.cpp with your source file name)
RUN g++ -std=c++11 -o my_program main.cpp -lOpenCL `pkg-config --cflags --libs opencv4`

# Set the command to run the program
CMD ["./my_program"]
