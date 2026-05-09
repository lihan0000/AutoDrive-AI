#include "YOLO_TRT.h"
#include <fstream>
#include <cuda_runtime_api.h>
#include <cmath>
#include <algorithm>

// TRT 日志记录器：过滤掉 Info 级别的冗余信息
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            // 可在此处打印警告或错误日志
            // std::cerr << "[TRT WARN/ERR] " << msg << std::endl;
        }
    }
} gLogger;

YOLOEngine::YOLOEngine(const std::string& engine_path) {
    // 1. 读取 Engine 文件二进制流
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[错误] 找不到模型文件！请检查路径: " << engine_path << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    // 2. 反序列化引擎
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    context = engine->createExecutionContext();
    delete[] trtModelStream;

    // 3. 创建异步 CUDA 流并分配 GPU 显存 (针对 YOLOv8/11 Segmentation 的结构)
    cudaStreamCreate(&stream);
    cudaMalloc(&buffers[0], 1 * 3 * 640 * 640 * sizeof(float));  // Input: 图像输入
    cudaMalloc(&buffers[1], 1 * 300 * 38 * sizeof(float));       // Output0: 检测框与掩码系数 (假设最多300个目标)
    cudaMalloc(&buffers[2], 1 * 32 * 160 * 160 * sizeof(float)); // Output1: 掩码原型 (Proto)
}

std::vector<Object> YOLOEngine::infer(cv::Mat& img) {
    std::vector<Object> results;

    // ==========================================
    // 1. 图像预处理 (LetterBox/Resize -> RGB -> 归一化 -> CHW)
    // ==========================================
    cv::Mat pr_img;
    cv::resize(img, pr_img, cv::Size(640, 640));
    cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
    pr_img.convertTo(pr_img, CV_32FC3, 1.0f / 255.0f);

    std::vector<float> input_data(3 * 640 * 640);
    std::vector<cv::Mat> chw;
    for (int i = 0; i < 3; ++i) {
        chw.push_back(cv::Mat(640, 640, CV_32FC1, input_data.data() + i * 640 * 640));
    }
    cv::split(pr_img, chw);

    // 将图像数据从 Host (CPU) 拷贝至 Device (GPU)
    cudaMemcpyAsync(buffers[0], input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    // ==========================================
    // 2. 执行 AI 推理 (TensorRT 10.x 专属语法)
    // ==========================================
    for (int i = 0; i < 3; ++i) {
        context->setTensorAddress(engine->getIOTensorName(i), buffers[i]);
    }
    context->enqueueV3(stream);

    // ==========================================
    // 3. 结果后处理 (拷贝回 CPU 并解析)
    // ==========================================
    std::vector<float> output_boxes(300 * 38);
    std::vector<float> output_proto(32 * 160 * 160);

    cudaMemcpyAsync(output_boxes.data(), buffers[1], output_boxes.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output_proto.data(), buffers[2], output_proto.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat proto_mat(32, 160 * 160, CV_32FC1, output_proto.data());

    float rx = (float)img.cols / 640.0f;
    float ry = (float)img.rows / 640.0f;

    for (int i = 0; i < 300; ++i) {
        float* row = output_boxes.data() + i * 38;
        float conf = row[4];
        
        // 置信度阈值过滤
        if (conf < 0.4f) continue;

        int cls = (int)row[5];
        float x1 = row[0];
        float y1 = row[1];
        float x2 = row[2];
        float y2 = row[3];

        // 边框还原到原图尺寸
        cv::Rect box;
        box.x = std::max(0, (int)std::round(x1 * rx));
        box.y = std::max(0, (int)std::round(y1 * ry));
        box.width = std::min(img.cols - box.x, (int)std::round((x2 - x1) * rx));
        box.height = std::min(img.rows - box.y, (int)std::round((y2 - y1) * ry));

        if (box.width <= 0 || box.height <= 0) continue;

        // 生成目标实例掩码 (Mask)
        cv::Mat mask_coeffs(1, 32, CV_32FC1, row + 6);
        cv::Mat mask_mat = mask_coeffs * proto_mat; 
        mask_mat = mask_mat.reshape(1, 160);        

        // Sigmoid 激活函数
        cv::exp(-mask_mat, mask_mat);
        mask_mat = 1.0f / (1.0f + mask_mat);

        // Resize 掩码到原图大小并二值化
        cv::Mat mask_resized;
        cv::resize(mask_mat, mask_resized, img.size());
        cv::Mat object_mask = mask_resized(box) > 0.5f; 

        // 放置回全尺寸空矩阵中
        cv::Mat full_mask = cv::Mat::zeros(img.size(), CV_8UC1);
        object_mask.copyTo(full_mask(box));

        Object obj;
        obj.rect = box;
        obj.label = cls;
        obj.prob = conf;
        obj.mask = full_mask;
        results.push_back(obj);
    }

    return results;
}

YOLOEngine::~YOLOEngine() {
    cudaStreamDestroy(stream);
    for (int i = 0; i < 3; i++) cudaFree(buffers[i]);
    if (context) delete context;
    if (engine)  delete engine;
    if (runtime) delete runtime;
}