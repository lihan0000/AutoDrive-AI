<div style="font-family: 'SimSun', 'STSong', '宋体', serif;">

# AutoDrive-AI: 自动驾驶边缘感知与控制系统全栈开发记录

---

## 🛠️ 项目演进与模块索引（按时间倒序排布）

| 阶段 | 核心功能模块 | 核心代码/路径 | 演示视频/文档 (点击下载) |
| :--- | :--- | :--- | :--- |
| **01** | **ROS2 端侧部署与视觉跟踪**<br>基于 Orin Nano 实现小车自动抓取、巡径，DeepStream GPU 加速。 | [📁 TEST_LEARNING](./TEST_LEARNING) | (学生项目已归档) |
| **02** | **K230 边缘端视觉与矿洞/渔场监测**<br>利用 K230 + CANMV 实现矿洞检测与渔场船只直线巡航。 | [📁 Visual_K230_Drive](./Visual_K230_Drive) | [📥 矿洞视频](./Visual_K230_Drive/Person/Person/PersonA_mp4.mp4?raw=true)<br>[📥 渔场视频](./Visual_K230_Drive/Vessel/Vessel/VesselsA_mp4.mp4?raw=true) |
| **03** | **无人机场景识别与 Transformer 预测**<br>手写 ENET/CNNs/Transformer 编码层，实现飞行意图判断。 | [📁 AutoDrive 模型](./UAV_Action_Prediction_Trial/AutoDrive) | [📥 实机视角](./UAV_Action_Prediction_Trial/AB_mp4.mp4?raw=true)<br>[📥 识别视角](./UAV_Action_Prediction_Trial/CD_mp4.mp4?raw=true) |
| **04** | **UM982 RTK 高精度定位配置**<br>完成基准站与移动站 4G 链路搭建，解决安装方向纠偏。 | [📁 UM982_RTK](./UM982_RTK) | [📄 配置文档](./UM982_RTK/UM982_RTK.docx?raw=true) |
| **05** | **RDKX5 深度相机与 IMU 滤波**<br>BMI088 IMU 误差分析与多种滤波算法实机测试。 | [📁 RDKX5_感知模块](./RDKX5_IMU_DEPTH_Camera) | (实机调试数据) |
| **06** | **PX6C 非标 CAN 协议 LUA 解析**<br>编写 LUA 脚本实现非标 CAN 控制协议的底层闭环。 | [📁 PX6C_Control](./Orin_Nano_PX6C/PX6C_Lua_Control) | (底层逻辑代码) |
| **07** | **深度相机与底盘闭环行走控制**<br>Orin Nano + Gemini 相机实现毫米级测距与目标跟随。 | [📁 Visual_Soild_D](./Orin_Nano_DEPTH_Camera_Distance/Visual_Soild_D/Visual_Soild_D) | (串口控制指令) |
| **08** | **双目仿生高速避障系统 (终极方案)**<br>提取视差/膨胀等仿生直觉，解决 DC 均衡编码与带宽噪点问题。 | [📁 避障核心源码](./Orin_Nano_PX6C) | [📥 避障刹停视频](./Orin_Nano_PX6C/Obstacle_Avoidance_Detour_TEST/LR_mp4.mp4?raw=true) |

---

### 💡 阅读提示
1. **排版对齐**：本页面采用表格化排版，以确保在宋体环境下各模块说明与路径严格对齐。
2. **视频下载**：点击“下载”链接后，GitHub 会打开原始文件页面，您可以点击页面上的 **"Download"** 按钮或直接使用浏览器“另存为”保存视频。
3. **技术路线**：项目从基础的 ROS2 巡径进化至复杂的双目仿生避障，重点攻克了高速信号传输中的链路稳定性问题。

</div>