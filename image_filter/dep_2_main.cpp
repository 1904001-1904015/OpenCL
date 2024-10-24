#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <CL/cl.h>
#include <sys/stat.h>
#include <ctime>

#define COMPUTE_KERNEL_FILENAME ("./kernal_code.cl")

int width = 512, height = 512;
cl_int err;
size_t global;
size_t local;
cl_device_id device_id;
cl_context context;
cl_command_queue commands;
cl_program program;
cl_kernel kernel;
cl_mem input;
cl_mem output;
int pixelCount = width * height;
int gpu = 1;
double avgTime = 0;
int counts = 0;

// Loads the OpenCL kernel source file
static int loadKernelSource(const char *file_name, char **result_string, size_t *string_len) {
    int fd;
    unsigned file_len;
    struct stat file_status;

    *string_len = 0;
    fd = open(file_name, O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening file " << file_name << std::endl;
        return -1;
    }

    fstat(fd, &file_status);
    file_len = (unsigned)file_status.st_size;
    *result_string = (char *)calloc(file_len + 1, sizeof(char));
    read(fd, *result_string, file_len);
    close(fd);
    *string_len = file_len;

    return 0;
}

// Initializes OpenCL platform, context, and command queue
int initializeOpenCL() {
    cl_platform_id *platform;
    cl_uint numplatform;
    err = clGetPlatformIDs(0, NULL, &numplatform);
    platform = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numplatform);
    err = clGetPlatformIDs(numplatform, platform, NULL);
    err = clGetDeviceIDs(platform[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: Failed to create a device group!\n";
            return EXIT_FAILURE;
        }
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);

    char *source = 0;
    size_t length = 0;
    loadKernelSource(COMPUTE_KERNEL_FILENAME, &source, &length);
    program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);

    err = clBuildProgram(program, 0, NULL, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[1024];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cerr << buffer << std::endl;
        exit(1);
    }

    kernel = clCreateKernel(program, "myFilter", &err);
    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * pixelCount, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * pixelCount, NULL, NULL);
    
    return 0;
}

// Executes the OpenCL kernel for the image filter
int executeOpenCL(uchar *inputData, uchar *outputData) {
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(uchar) * pixelCount, inputData, 0, NULL, NULL);
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    global = pixelCount;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(commands);
    clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(uchar) * pixelCount, outputData, 0, NULL, NULL);

    return 0;
}

// Releases OpenCL resources
void releaseOpenCLResources() {
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}

// Reads a PPM image from file
std::vector<unsigned char> readPPM(const std::string &filename, int &width, int &height) {
    std::ifstream file(filename, std::ios::binary);
    std::string header;
    file >> header;

    if (header != "P6") {
        std::cerr << "Unsupported PPM format" << std::endl;
        exit(1);
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.ignore(256, '\n');

    std::vector<unsigned char> data(width * height * 3);
    file.read(reinterpret_cast<char*>(data.data()), data.size());

    return data;
}

// Saves a PPM image to file
void writePPM(const std::string &filename, const std::vector<unsigned char> &data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

// Converts an RGB image to grayscale
std::vector<unsigned char> convertToGrayscale(const std::vector<unsigned char> &rgbImage, int width, int height) {
    std::vector<unsigned char> grayImage(width * height);
    for (int i = 0; i < width * height; ++i) {
        int r = rgbImage[3 * i];
        int g = rgbImage[3 * i + 1];
        int b = rgbImage[3 * i + 2];
        grayImage[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return grayImage;
}

// Applies a manual filter using a provided kernel matrix
uchar *applyManualFilter(uchar *data, uchar *result, float filter[][3], int kernel_height = 3, int kernel_width = 3, int height = 512, int width = 512) {
    uchar **extend_data = new uchar *[height + kernel_height - 1];
    for (int i = 0; i < height + kernel_height - 1; i++)
        extend_data[i] = new uchar[width + kernel_width - 1];

    for (int i = 0; i < height + kernel_height - 1; i++)
        for (int j = 0; j < width + kernel_width - 1; j++)
            extend_data[i][j] = 0;

    for (int i = kernel_height / 2, m = 0; i < kernel_height / 2 + height && m < height; i++, m++)
        for (int j = kernel_width / 2, n = 0; j < kernel_width / 2 + width && n < width; j++, n++)
            extend_data[i][j] = data[m * width + n];

    for (int i = kernel_height / 2; i < kernel_height / 2 + height; i++) {
        for (int j = kernel_width / 2; j < kernel_width / 2 + width; j++) {
            float sum = 0;
            for (int m = -kernel_height / 2; m <= kernel_height / 2; m++) {
                for (int n = -kernel_width / 2; n <= kernel_width / 2; n++) {
                    sum += extend_data[i + m][j + n] * filter[m + kernel_height / 2][n + kernel_width / 2];
                }
            }
            result[(i - kernel_height / 2) * width + j - kernel_width / 2] = uchar(sum);
        }
    }

    for (int i = 0; i < height + kernel_height - 1; i++)
        delete[] extend_data[i];
    delete[] extend_data;

    return result;
}

int main() {
    int width, height;
    std::vector<unsigned char> rgbImage = readPPM("./input/cat.ppm", width, height);

    std::vector<unsigned char> grayImage = convertToGrayscale(rgbImage, width, height);

    initializeOpenCL();

    float filter[3][3] = {
        {-1, 0, -1},
        {0, 4, 0},
        {-1, 0, -1}
    };

    uchar *result = new uchar[pixelCount];
    std::vector<uchar> outputData(pixelCount, 0);

    uchar *data = grayImage.data();

    std::clock_t start1 = std::clock();
    executeOpenCL(data, outputData.data());
    double duration1 = (std::clock() - start1) / (double)CLOCKS_PER_SEC;
    avgTime += duration1;
    counts++;
    std::cout << "OpenCL: time: " << duration1 << " avgTime: " << avgTime / counts << '\n';

    releaseOpenCLResources();

    std::clock_t start2 = std::clock();
    applyManualFilter(data, result, filter, 3, 3, height, width);
    double duration2 = (std::clock() - start2) / (double)CLOCKS_PER_SEC;
    avgTime += duration2;
    counts++;
    std::cout << "Manual: time: " << duration2 << " avgTime: " << avgTime / counts << '\n';

    std::vector<unsigned char> outputImage(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        outputImage[3 * i] = result[i];
        outputImage[3 * i + 1] = result[i];
        outputImage[3 * i + 2] = result[i];
    }

    writePPM("./result/output.ppm", outputImage, width, height);

    delete[] result;
    return 0;
}
