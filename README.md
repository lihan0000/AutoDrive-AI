# AutoDrive-AI: 自动驾驶边缘感知与控制系统全栈架构

[![Status](https://img.shields.io/badge/状态-稳定运行-success)](#) [![Tech](https://img.shields.io/badge/核心技术栈-C++%20%7C%20CUDA%20%7C%20ROS2-blue)](#) [![Hardware](https://img.shields.io/badge/硬件平台-Orin_Nano%20%7C%20K230%20%7C%20RDKX5-orange)](#) 

> **项目概述**：本项目记录了从端侧视觉部署到多模态异构融合的完整演进过程。系统涵盖了从顶层 Transformer 动作预测、ROS2 导航，到底层 IMU 误差滤波、非标 CAN 协议解析，以及高速物理链路（DC均衡/带宽优化）的全栈工程实践。

---

## 🛠️ 项目演进与核心模块索引

### 01. ROS2 端侧部署与视觉跟踪 (基础构建)
* **核心功能**：基于 Nvidia Jetson Orin Nano 与 ROS2 实现小车自动抓取、巡径行走与动作标识。利用 DeepStream 压榨 GPU 加速性能，在本地验证多种视觉跟踪算法、Transformer 动作预测模型及端侧部署。
* 📁 **代码目录**：[👉 点击跳转：TEST_LEARNING](./TEST_LEARNING)

### 02. K230 边缘端视觉与 CANMV 控制 (场景落地)
* **核心功能**：利用 K230 芯片配合 CANMV，在特定工业场景下实现矿洞安全检测，以及渔场作业船只的视觉直线自动巡航。
* 🎥 **实况视频**：
  * [📥 点击下载：矿洞安全检测实测视频](./Visual_K230_Drive/Person/Person/PersonA_mp4.mp4?raw=true)
  * [📥 点击下载：渔场作业船只自动巡航视频](./Visual_K230_Drive/Vessel/Vessel/VesselsA_mp4.mp4?raw=true)
* 📁 **代码目录**：[👉 点击跳转：Visual_K230_Drive](./Visual_K230_Drive)

### 03. 无人机场景识别与 Transformer 预测 (算法进阶)
* **核心功能**：脱离开源框架，手写 ENET、CNNs 及 Transformer 编码层代码。成功应用于无人机飞行过程中的实时场景识别与动作意图预测。
* 🎥 **实况视频**：
  * [📥 点击下载：无人机实机飞行视角](./UAV_Action_Prediction_Trial/AB_mp4.mp4?raw=true)
  * [📥 点击下载：底层场景识别与预测视角](./UAV_Action_Prediction_Trial/CD_mp4.mp4?raw=true)
* 📁 **模型结构**：[👉 点击跳转：UAV_Action_Prediction_Trial/AutoDrive](./UAV_Action_Prediction_Trial/AutoDrive)

### 04. UM982 RTK 高精度定位配置 (绝对观测引入)
* **核心功能**：完成 UM982 RTK 基准站与移动站的完整 4G 链路搭建与配置，解决安装方向纠偏等工程问题，为系统提供厘米级绝对定位基准。
* 📄 **配置文档**：[👉 点击查看：UM982_RTK.docx](./UM982_RTK/UM982_RTK.docx)

### 05. RDKX5 深度相机与 IMU 滤波 (底层状态估计)
* **核心功能**：基于 RDKX5 平台驱动 Gemini 深度相机与 BMI088 IMU。深入分析 IMU 积分误差积累的物理特性，实机测试多种滤波方案，实现稳定姿态估计。
* 📁 **代码目录**：[👉 点击跳转：RDKX5_IMU_DEPTH_Camera](./RDKX5_IMU_DEPTH_Camera)

### 06. PX6C 非标 CAN 协议解析 (底盘控制)
* **核心功能**：编写 PX6C 的 LUA 底层控制代码，打通底层硬件执行逻辑，实现非标 CAN 控制协议的精准解析与下发。
* 📁 **代码目录**：[👉 点击跳转：Orin_Nano_PX6C/PX6C_Lua_Control](./Orin_Nano_PX6C/PX6C_Lua_Control)

### 07. 深度相机与底盘闭环控制 (感知与控制协同)
* **核心功能**：结合 Jetson Orin Nano 与 Gemini 深度相机，实现毫米级精度的目标测距。配合 PX6C 串口指令，完成真实场景下的目标跟随与自动行驶闭环。
* 📁 **代码目录**：[👉 点击跳转：Orin_Nano_DEPTH_Camera_Distance](./Orin_Nano_DEPTH_Camera_Distance/Visual_Soild_D/Visual_Soild_D)

### 08. 双目仿生高速避障系统 (终极融合与系统级降级)
* **核心功能**：利用双摄架构，结合 YOLO 26 模型，提取压线、膨胀、视差、方差等仿生直觉信息。完美实现低成本、低延迟的高速避障、绕行与紧急刹停。
* **物理层突破**：**成功解决高速视频流传输中的 DC 均衡编码断链问题，以及恶劣环境噪点导致的带宽激增溢出问题，保障物理链路绝对稳定。**
* 🎥 **实况视频**：
  * [📥 点击下载：实机避障、绕行与紧急刹停测试](./Orin_Nano_PX6C/Obstacle_Avoidance_Detour_TEST/LR_mp4.mp4?raw=true)

---
*注：本项目重点展示从算法推理到底层硬件调度的全栈工程落地能力。点击上述视频链接可直接下载原文件查看实测效果。*