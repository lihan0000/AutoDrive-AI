cmake_minimum_required(VERSION 3.10)
project(PastureBot)

set(CMAKE_CXX_STANDARD 17)

# 1. 寻找系统级核心依赖库
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# 2. 定位头文件 (包含当前目录、奥比中光相机、ARM64系统库、OpenCV及CUDA)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libobsensor
    /usr/include/aarch64-linux-gnu
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# 3. 定位预编译的 ARM64 动态链接库目录
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/arm64)

# 4. 编译可执行程序 (注意源文件命名已更改为大写)
add_executable(bot_app main_ob.cpp YOLO_TRT.cpp)

# 5. 链接所有底层动态库 (顺序：OpenCV -> TensorRT -> CUDA -> 相机SDK)
target_link_libraries(bot_app 
    ${OpenCV_LIBS} 
    nvinfer        # TensorRT 核心库
    cudart         # CUDA 运行时库
    OrbbecSDK      # 奥比中光相机核心库 (libOrbbecSDK.so)
    depthengine    # 深度计算引擎库 (libdepthengine.so)
)