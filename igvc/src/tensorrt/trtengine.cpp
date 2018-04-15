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

static ICudaEngine* engine;
static IExecutionContext *context;


static float *inputDataHost, *outputDataHost;
static size_t numInput, numOutput, numInSize, numOutSize;

static Dims inputDims, outputDims;
static int inputWidth, inputHeight;
    
static int inputBindingIndex, outputBindingIndex;
    
static void *bindings[2];
static float *inputDataDevice, *outputDataDevice;

using namespace std::chrono;
    

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    engine = builder->buildCudaEngine(*network);

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}

int loadEngine()
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

    std::cout << "TEST1" << std::endl;

    /* get size of engine file */
    std::ifstream infile1("test.engine", std::ifstream::ate | std::ifstream::binary);
    if (!infile1.is_open()) {
        fprintf(stderr, "fail to open file to read: %s\n", "test.engine");
        return -1;
    }
    long fsize = infile1.tellg();
    infile1.close();

    std::cout << "TEST2" << std::endl;

    /* load engine file */
    std::ifstream infile("test.engine", std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        fprintf(stderr, "fail to open file to read: %s\n", "test.engine");
        return -1;
    }

    std::cout << "TEST3" << std::endl;
    char* p = new char[fsize];
    infile.read(p, fsize);
    infile.close();
    IRuntime* runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine((unsigned char*)p, fsize, nullptr);

    /* create execution context */
	context = engine->createExecutionContext();
    
    /* get the input / output dimensions */
    inputBindingIndex = engine->getBindingIndex("Placeholder");
    outputBindingIndex = engine->getBindingIndex("upsample3/BiasAdd");

    if (inputBindingIndex < 0)
    {
        std::cout << "Invalid input name." << std::endl;
        return EXIT_FAILURE;
    }

    if (outputBindingIndex < 0)
    {
        std::cout << "Invalid output name." << std::endl;
        return EXIT_FAILURE;
    }

    inputDims = engine->getBindingDimensions(inputBindingIndex);
    outputDims = engine->getBindingDimensions(outputBindingIndex);
    inputHeight = inputDims.d[1];
    inputWidth = inputDims.d[2];
    
    numInput = numTensorElements(inputDims);
    numOutput = numTensorElements(outputDims);
    
    numInSize = numInput * sizeof(float);    
    numOutSize = numOutput * sizeof(float);
        
    inputDataHost = (float*) malloc(numInput * sizeof(float));
    outputDataHost = (float*) malloc(numOutput * sizeof(float));
 
    cudaMalloc((void**)&inputDataDevice, numInput * sizeof(float));
    cudaMalloc((void**)&outputDataDevice, numOutput * sizeof(float));

    bindings[inputBindingIndex] = (void*) inputDataDevice;
    bindings[outputBindingIndex] = (void*) outputDataDevice;
    
    preprocessVgg(inputDataHost, inputDims);
   
    /* we need to keep the memory created by the parser */
    parser->destroy();

    return EXIT_SUCCESS;
}


cv::Mat executeEngine(cv::Mat image)
{

    /* read image, convert color, and resize */
    if (image.data == NULL)
    {
        std::cout << "Could not read image from file." << std::endl;
        return image;
    }    
    
    high_resolution_clock::time_point t1;
    high_resolution_clock::time_point t2;    
    t1 = high_resolution_clock::now();
    
    cv::resize(image, image, cv::Size(inputWidth, inputHeight));

    /* convert from uint8+NHWC to float+NCHW */
    cvImageToTensor(image, inputDataHost, inputDims);

    /* transfer to device */
    cudaMemcpy(inputDataDevice, inputDataHost, numInSize, cudaMemcpyHostToDevice);

    /* execute engine */
    context->execute(1, bindings);

    /* transfer output back to host */
    cudaMemcpy(outputDataHost, outputDataDevice, numOutSize, cudaMemcpyDeviceToHost);

    /* convert tensor to output image */    
    cv::Mat outImage(image.size(), image.type());
    cvTensorToImage(image, outputDataHost, inputDims);

    t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;
    
    return image;
}
