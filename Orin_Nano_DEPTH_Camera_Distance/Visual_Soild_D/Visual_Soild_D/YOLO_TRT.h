#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

// ==========================================
// 数据结构：存储单个目标的感知信息
// ==========================================
struct Object {
    cv::Rect rect;       // 目标的 2D 边界框 (Bounding Box)
    int label;           // 类别 ID (如：0代表牛，1代表栏杆)
    float prob;          // 预测置信度
    cv::Mat mask;        // 目标的二值化实例掩码 (尺寸已还原至原图分辨率)
};

// ==========================================
// 核心类：YOLO TensorRT 推理引擎
// ==========================================
class YOLOEngine {
public:
    // 构造函数：加载 .engine 模型并分配 GPU 显存
    YOLOEngine(const std::string& engine_path);
    
    // 析构函数：释放 GPU 资源和引擎对象
    ~YOLOEngine();
    
    // 核心推理函数：输入 OpenCV 图像，输出目标数组
    std::vector<Object> infer(cv::Mat& img);

private:
    nvinfer1::IRuntime* runtime;          // TRT 运行时
    nvinfer1::ICudaEngine* engine;        // TRT 序列化引擎
    nvinfer1::IExecutionContext* context; // 推理上下文
    
    void* buffers[3];                     // GPU 显存指针数组 (0:输入, 1:检测框输出, 2:掩码原型输出)
    cudaStream_t stream;                  // CUDA 异步流
};