#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "libobsensor/ObSensor.hpp"
#include "YOLO_TRT.h" // 统一大写

// ====== 串口通信需要的 Linux 头文件 ======
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>

// ==========================================
// 辅助功能：初始化串口通信
// 波特率: 115200, 数据位: 8, 校验位: N, 停止位: 1
// ==========================================
int init_serial(const char* portname) {
    int fd = open(portname, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        std::cerr << "[硬件警告] 无法打开串口 " << portname << "。请检查 sudo 权限或连线！" << std::endl;
        return -1;
    }
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) return -1;
    
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; 
    tty.c_iflag &= ~IGNBRK; 
    tty.c_lflag = 0; 
    tty.c_oflag = 0; 
    tty.c_cc[VMIN]  = 0; 
    tty.c_cc[VTIME] = 1; // 0.1秒超时
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); 
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD); 
    tty.c_cflag &= ~CSTOPB; 
    tty.c_cflag &= ~CRTSCTS;
    
    tcsetattr(fd, TCSANOW, &tty);
    return fd;
}

int main(int argc, char **argv) {
    // 建立硬件串口连接
    int serial_fd = init_serial("/dev/ttyTHS1");

    try {
        // ==========================================
        // 1. 初始化奥比中光深度相机 (RGB-D 对齐模式)
        // ==========================================
        ob::Pipeline pipeline;
        auto config = std::make_shared<ob::Config>();
        auto colorProfile = pipeline.getStreamProfileList(OB_SENSOR_COLOR)->getProfile(0);
        auto depthProfile = pipeline.getStreamProfileList(OB_SENSOR_DEPTH)->getProfile(0);
        
        config->enableStream(colorProfile);
        config->enableStream(depthProfile);
        config->setAlignMode(ALIGN_D2C_HW_MODE); // 硬件级深度对齐到彩色
        pipeline.start(config);

        // 获取相机内参（用于将像素坐标系转换为真实空间三维坐标）
        auto cameraParam = pipeline.getCameraParam();
        float fx = cameraParam.depthIntrinsic.fx;
        float cx = cameraParam.depthIntrinsic.cx;

        // ==========================================
        // 2. 初始化 AI 大脑 (YOLO TRT)
        // ==========================================
        std::cout << "[INFO] 正在加载 YOLO TensorRT 引擎..." << std::endl;
        YOLOEngine yolo_model("my_pasture.engine"); 
        std::cout << "[INFO] 初始化完成！按 ESC 键退出程序。" << std::endl;

        // ==========================================
        // 3. 主循环：捕捉视频流 -> 推理 -> 计算深度 -> 发送串口
        // ==========================================
        while (true) {
            auto frameSet = pipeline.waitForFrames(100);
            if (!frameSet || !frameSet->colorFrame() || !frameSet->depthFrame()) continue;

            auto colorFrame = frameSet->colorFrame();
            auto depthFrame = frameSet->depthFrame();

            // 解析彩色图像
            cv::Mat color_mat;
            if (colorFrame->format() == OB_FORMAT_MJPG) {
                cv::Mat rawMat(1, colorFrame->dataSize(), CV_8UC1, colorFrame->data());
                color_mat = cv::imdecode(rawMat, cv::IMREAD_COLOR);
            } else if (colorFrame->format() == OB_FORMAT_RGB || colorFrame->format() == OB_FORMAT_RGB888) {
                cv::Mat raw(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
                cv::cvtColor(raw, color_mat, cv::COLOR_RGB2BGR);
            }
            if (color_mat.empty()) continue; 

            // 解析深度图像 (16位毫米值)
            cv::Mat depth_mat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());

            // 运行 YOLO 推理
            std::vector<Object> objs = yolo_model.infer(color_mat);

            long long total_Z = 0;
            int valid_obj_count = 0;

            // 遍历所有被识别的目标
            for (auto& obj : objs) {
                std::vector<uint16_t> valid_depths;

                // 基于实例掩码 (Mask)，提取目标表面的所有有效深度像素
                for (int y = obj.rect.y; y < obj.rect.y + obj.rect.height; y++) {
                    for (int x = obj.rect.x; x < obj.rect.x + obj.rect.width; x++) {
                        // 越界保护
                        if (x < 0 || x >= color_mat.cols || y < 0 || y >= color_mat.rows) continue;
                        
                        // 仅当掩码像素有效时提取深度
                        if (!obj.mask.empty() && obj.mask.at<uchar>(y, x) > 0) {
                            uint16_t d = depth_mat.at<uint16_t>(y, x);
                            if (d > 0 && d < 10000) { // 过滤 0 和极端距离
                                valid_depths.push_back(d);
                            }
                        }
                    }
                }

                // 计算该目标的代表性深度 (利用中位数滤波抵御噪点)
                if (!valid_depths.empty()) {
                    std::nth_element(valid_depths.begin(), valid_depths.begin() + valid_depths.size() / 2, valid_depths.end());
                    uint16_t target_Z = valid_depths[valid_depths.size() / 2];

                    total_Z += target_Z;
                    valid_obj_count++;

                    // 换算真实世界的 X 坐标
                    float u = obj.rect.x + obj.rect.width / 2.0f;
                    float target_X = (u - cx) * target_Z / fx;

                    // UI 渲染：绘制半透明掩码、检测框和距离文本
                    cv::Mat color_mask(color_mat.size(), CV_8UC3, cv::Scalar(0, 0, 255)); 
                    cv::Mat masked_roi;
                    cv::bitwise_and(color_mask, color_mask, masked_roi, obj.mask); 
                    cv::addWeighted(color_mat, 1.0, masked_roi, 0.5, 0.0, color_mat); 

                    cv::rectangle(color_mat, obj.rect, cv::Scalar(0, 255, 0), 2);
                    std::string text = "Z:" + std::to_string(target_Z) + "mm X:" + std::to_string((int)target_X) + "mm";
                    cv::putText(color_mat, text, cv::Point(obj.rect.x, std::max(0, obj.rect.y - 10)),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                }
            }

            // ==========================================
            // 4. 串口通信：计算全局平均距离并下发指令
            // ==========================================
            if (valid_obj_count > 0) {
                uint16_t avg_Z = total_Z / valid_obj_count;
                
                // 在 UI 上显示即将发出的指令值
                cv::putText(color_mat, "AVG Z: " + std::to_string(avg_Z) + " mm", cv::Point(15, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

                // 发送数据，例如 "Z:1250\n"
                if (serial_fd >= 0) {
                    std::string msg = "Z:" + std::to_string(avg_Z) + "\n";
                    write(serial_fd, msg.c_str(), msg.length());
                }
            }

            cv::imshow("Pasture View", color_mat);
            if (cv::waitKey(1) == 27) break; // ESC 退出
        }
        
        // 5. 资源释放
        pipeline.stop();
        if (serial_fd >= 0) close(serial_fd);

    } catch (ob::Error &e) {
        std::cerr << "[相机故障] 报错信息: " << e.getName() << "\n" << e.getMessage() << std::endl;
    }
    return 0;
}