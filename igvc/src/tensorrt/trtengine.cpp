#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include <opencv2/opencv.hpp>
#include "utils.h"

#include "NvUtils.h"

#include <chrono>

using namespace nvuffparser;
using namespace nvinfer1;
#include "common.h"

static Logger gLogger;

#define MAX_WORKSPACE (1 << 30)

static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}


int loadAndExecuteEngine()
{
    auto parser = createUffParser();
    parser->registerInput("Placeholder", DimsCHW(3, 340, 450));
    parser->registerOutput("upsample3/BiasAdd");


    /* create and save engine file from uff file */
//    int maxBatchSize = 1;
//    ICudaEngine* engine = loadModelAndCreateEngine("/home/nvidia/Desktop/test.uff", maxBatchSize, parser);

//    if (!engine)
//        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");

//    nvinfer1::IHostMemory* gieModelStream = engine->serialize(); // GIE model
//    fprintf(stdout, "allocate memory size: %d bytes\n", gieModelStream->size());
//    std::ofstream outfile("test.engine", std::ios::out | std::ios::binary);
//    if (!outfile.is_open()) {
//        fprintf(stderr, "fail to open file to write: %s\n", "test.engine");
//        return -1;
//    }
//    unsigned char* p = (unsigned char*)gieModelStream->data();
//    outfile.write((char*)p, gieModelStream->size());
//    outfile.close();

    /* get size of engine file */
    std::ifstream infile1("test.engine", std::ifstream::ate | std::ifstream::binary);
    long fsize = infile1.tellg();
    infile1.close();

    /* load engine file */
    std::ifstream infile("test.engine", std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        fprintf(stderr, "fail to open file to write: %s\n", "test.engine");
        return -1;
    }
    char* p = new char[fsize];
    infile.read(p, fsize);
    infile.close();
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine((unsigned char*)p, fsize, nullptr);

    /* create execution context */
	IExecutionContext *context = engine->createExecutionContext();

    /* get the input / output dimensions */
    int inputBindingIndex, outputBindingIndex;
    inputBindingIndex = engine->getBindingIndex("Placeholder");
    outputBindingIndex = engine->getBindingIndex("upsample3/BiasAdd");

    if (inputBindingIndex < 0)
    {
        std::cout << "Invalid input name." << std::endl;
        return 1;
    }

    if (outputBindingIndex < 0)
    {
        std::cout << "Invalid output name." << std::endl;
        return 1;
    }

    Dims inputDims, outputDims;
    inputDims = engine->getBindingDimensions(inputBindingIndex);
    outputDims = engine->getBindingDimensions(outputBindingIndex);
    int inputWidth, inputHeight;
    inputHeight = inputDims.d[1];
    inputWidth = inputDims.d[2];

    /* read image, convert color, and resize */
    std::cout << "Preprocessing input..." << std::endl;
    cv::Mat image = cv::imread("/home/nvidia/Desktop/test.jpg", CV_LOAD_IMAGE_COLOR);

    if (image.data == NULL)
    {
        std::cout << "Could not read image from file." << std::endl;
        return 1;
    }

    cv::resize(image, image, cv::Size(inputWidth, inputHeight));
    cv::imshow("TEST", image);
    cv::waitKey(0);

    using namespace std::chrono;
    /* convert from uint8+NHWC to float+NCHW */
    float *inputDataHost, *outputDataHost;
    size_t numInput, numOutput;
    numInput = numTensorElements(inputDims);
    numOutput = numTensorElements(outputDims);
    inputDataHost = (float*) malloc(numInput * sizeof(float));
    outputDataHost = (float*) malloc(numOutput * sizeof(float));
    cvImageToTensor(image, inputDataHost, inputDims);

    preprocessVgg(inputDataHost, inputDims);

    /* transfer to device */
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    float *inputDataDevice, *outputDataDevice;
    cudaMalloc((void**)&inputDataDevice, numInput * sizeof(float));
    cudaMalloc((void**)&outputDataDevice, numOutput * sizeof(float));
    cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);
    void *bindings[2];
    bindings[inputBindingIndex] = (void*) inputDataDevice;
    bindings[outputBindingIndex] = (void*) outputDataDevice;

    /* execute engine */
    std::cout << "Executing inference engine..." << std::endl;
    const int kBatchSize = 1;
    context->execute(kBatchSize, bindings);

    /* transfer output back to host */
    cudaMemcpy(outputDataHost, outputDataDevice, numOutput * sizeof(float), cudaMemcpyDeviceToHost);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;

    cv::Mat outImage(image.size(), image.type());
    cvTensorToImage(image, outputDataHost, inputDims);
    cv::imshow("TEST", image);
    cv::waitKey(0);

    /* we need to keep the memory created by the parser */
    parser->destroy();

    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
