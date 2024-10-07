# OpenCL Integer Cubic Calculator

This project demonstrates the use of OpenCL to compute the cubic values of an array of integers on both macOS and Linux.

## Prerequisites

### macOS
1. Install **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. OpenCL is pre-installed on macOS, so no additional setup is required for it.

### Linux
1. Install OpenCL drivers and utilities:
   - For **Intel GPUs**:
     ```bash
     sudo apt install intel-opencl-icd
     ```
   - For **AMD GPUs**:
     ```bash
     sudo apt install mesa-opencl-icd
     ```
   - For **NVIDIA GPUs**:
     ```bash
     sudo apt install nvidia-opencl-dev opencl-headers
     ```

2. Install the necessary OpenCL libraries:
   ```bash
   sudo apt install ocl-icd-libopencl1 clinfo
   ```

## How to Run the Program

1. Clone the repository:
   ```bash
   git clone https://github.com/1904001-1904015/OpenCLCubeCalculator.git
   cd OpenCLCubeCalculator
   ```

2. Run the `command.sh` script:
   ```bash
   ./command.sh your_program_name.c
   ```

This script will automatically detect your operating system (macOS or Linux), compile the program, and run it.
