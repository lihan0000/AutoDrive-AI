#include <iostream>
#include <opencv2/opencv.hpp>
#include "libobsensor/ObSensor.hpp"

int main(int argc, char **argv) {
    try {
        ob::Context ctx;
        ctx.setLoggerSeverity(OB_LOG_SEVERITY_WARN);
        ob::Pipeline pipe;

        std::cout << "正在读取相机配置..." << std::endl;
        pipe.start(nullptr); 
        std::cout << "🚀 相机底层启动成功，正在疯狂抓拍..." << std::endl;

        int saved_count = 0;

        // 我们只抓拍 5 组照片，拍完自动退出
        while (saved_count < 5) { 
            auto frameSet = pipe.waitForFrames(100);
            if (frameSet == nullptr) continue;

            auto colorFrame = frameSet->colorFrame();
            auto depthFrame = frameSet->depthFrame();

            bool color_saved = false;
            bool depth_saved = false;

            // --- 处理彩色图 ---
            if (colorFrame) {
                if (colorFrame->format() == OB_FORMAT_RGB || colorFrame->format() == OB_FORMAT_RGB888) {
                    cv::Mat colorMat(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
                    cv::Mat bgrMat;
                    cv::cvtColor(colorMat, bgrMat, cv::COLOR_RGB2BGR);
                    cv::imwrite("color_" + std::to_string(saved_count) + ".png", bgrMat);
                    color_saved = true;
                } else if (colorFrame->format() == OB_FORMAT_MJPG) {
                    cv::Mat rawMat(1, colorFrame->dataSize(), CV_8UC1, colorFrame->data());
                    cv::Mat colorMat = cv::imdecode(rawMat, cv::IMREAD_COLOR);
                    if (!colorMat.empty()) {
                        cv::imwrite("color_" + std::to_string(saved_count) + ".png", colorMat);
                        color_saved = true;
                    }
                }
            }

            // --- 处理深度图 ---
            if (depthFrame) {
                cv::Mat depthMat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());
                cv::Mat depthShow;
                // 涂色处理
                cv::normalize(depthMat, depthShow, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::applyColorMap(depthShow, depthShow, cv::COLORMAP_JET);
                cv::imwrite("depth_" + std::to_string(saved_count) + ".png", depthShow);
                depth_saved = true;
            }

            if (color_saved || depth_saved) {
                std::cout << "✅ 成功保存第 " << saved_count + 1 << " 组彩色与深度照片！" << std::endl;
                saved_count++;
            }
        }

        pipe.stop();
        std::cout << "🎉 测试完毕，完美退出！照片已保存在 build 文件夹下。" << std::endl;

    } catch (ob::Error &e) {
        std::cerr << "❌ 错误: " << e.getMessage() << std::endl;
    }

    return 0;
}
