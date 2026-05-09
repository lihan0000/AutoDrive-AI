#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>

class FastGPIO {
private:
    int fd_;
public:
    FastGPIO(int sysfs_gpio_num) {
        std::string gpio_dir = "/sys/class/gpio/gpio" + std::to_string(sysfs_gpio_num);
        if (access(gpio_dir.c_str(), F_OK) != 0) {
            std::ofstream export_file("/sys/class/gpio/export");
            export_file << sysfs_gpio_num;
            export_file.close();
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
        }
        std::ofstream dir_file(gpio_dir + "/direction");
        dir_file << "out";
        dir_file.close();
        fd_ = open((gpio_dir + "/value").c_str(), O_WRONLY);
        
        setLow(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); setHigh(); 
    }
    ~FastGPIO() { if (fd_ >= 0) close(fd_); }
    void setLow() { write(fd_, "0", 1); }   
    void setHigh() { write(fd_, "1", 1); }  
};

class BMI088 {
private:
    int spi_fd_;
    FastGPIO acc_cs_;
    FastGPIO gyro_cs_;

    void writeReg(uint8_t reg, uint8_t value, FastGPIO& cs) {
        cs.setLow(); 
        uint8_t tx[2] = {static_cast<uint8_t>(reg & 0x7F), value};
        struct spi_ioc_transfer tr = {0};
        tr.tx_buf = (unsigned long)tx; tr.len = 2; tr.speed_hz = 5000000;
        ioctl(spi_fd_, SPI_IOC_MESSAGE(1), &tr);
        cs.setHigh(); 
    }

public:
    BMI088(int acc_gpio_num, int gyro_gpio_num) 
        : acc_cs_(acc_gpio_num), gyro_cs_(gyro_gpio_num) {
        spi_fd_ = open("/dev/spidev1.0", O_RDWR);
        uint8_t mode = SPI_MODE_0; uint32_t speed = 5000000;
        ioctl(spi_fd_, SPI_IOC_WR_MODE, &mode);
        ioctl(spi_fd_, SPI_IOC_WR_MAX_SPEED_HZ, &speed);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ~BMI088() { if (spi_fd_ >= 0) close(spi_fd_); }

    void initialize() {
        // --- 加速度计配置 ---
        writeReg(0x7D, 0x04, acc_cs_); 
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        writeReg(0x7C, 0x00, acc_cs_); 
        
        // --- 陀螺仪核心配置 (查阅博世手册) ---
        writeReg(0x15, 0x00, gyro_cs_); 
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // 🚀 硬件级优化：修改陀螺仪量程 (0x0F 寄存器)
        // 0x00: 2000 dps | 0x01: 1000 dps | 0x02: 500 dps
        writeReg(0x0F, 0x01, gyro_cs_); // 压缩到 1000 dps，灵敏度翻倍！
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    void readAccelBurst(int16_t &x, int16_t &y, int16_t &z) {
        acc_cs_.setLow();
        uint8_t tx[8] = {0x12 | 0x80, 0, 0, 0, 0, 0, 0, 0};
        uint8_t rx[8] = {0};
        struct spi_ioc_transfer tr = {0};
        tr.tx_buf = (unsigned long)tx; tr.rx_buf = (unsigned long)rx;
        tr.len = 8; tr.speed_hz = 5000000;
        ioctl(spi_fd_, SPI_IOC_MESSAGE(1), &tr);
        acc_cs_.setHigh();
        x = (rx[3] << 8) | rx[2]; y = (rx[5] << 8) | rx[4]; z = (rx[7] << 8) | rx[6];
    }

    void readGyroBurst(int16_t &x, int16_t &y, int16_t &z) {
        gyro_cs_.setLow();
        uint8_t tx[7] = {0x02 | 0x80, 0, 0, 0, 0, 0, 0};
        uint8_t rx[7] = {0};
        struct spi_ioc_transfer tr = {0};
        tr.tx_buf = (unsigned long)tx; tr.rx_buf = (unsigned long)rx;
        tr.len = 7; tr.speed_hz = 5000000;
        ioctl(spi_fd_, SPI_IOC_MESSAGE(1), &tr);
        gyro_cs_.setHigh();
        x = (rx[2] << 8) | rx[1]; y = (rx[4] << 8) | rx[3]; z = (rx[6] << 8) | rx[5];
    }
};

int main() {
    int sys_gpio_acc = 394;   
    int sys_gpio_gyro = 396;  

    std::cout << "正在初始化 BMI088..." << std::endl;
    BMI088 imu(sys_gpio_acc, sys_gpio_gyro);
    imu.initialize();

    // 加速度比例：默认 +-6g
    double acc_scale = (6.0 / 32768.0) * 9.80665;
    // 陀螺仪比例：我们已经在硬件里把它改成了 +-1000 dps！
    double gyro_scale = (1000.0 / 32768.0) * (3.1415926535 / 180.0);

    // ==========================================
    // 🚀 软件级优化：开机静止零偏校准 (Zero-Bias)
    // ==========================================
    double bgx = 0, bgy = 0, bgz = 0;
    std::cout << "\n⚠️ 准备校准！请将地瓜派平放桌面，绝对不要触碰它！" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2)); // 给用户两秒钟松手
    
    std::cout << "⏳ 正在进行高精度零偏采样 (耗时 1 秒)..." << std::endl;
    int calib_count = 200;
    for (int i = 0; i < calib_count; ++i) {
        int16_t raw_gx, raw_gy, raw_gz;
        imu.readGyroBurst(raw_gx, raw_gy, raw_gz);
        bgx += raw_gx * gyro_scale;
        bgy += raw_gy * gyro_scale;
        bgz += raw_gz * gyro_scale;
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 200Hz 采样
    }
    bgx /= calib_count; bgy /= calib_count; bgz /= calib_count;
    std::printf("✅ 校准完美结束！测量到的静止漂移为: X:%.4f Y:%.4f Z:%.4f\n\n", bgx, bgy, bgz);

    std::cout << "🚀 开始输出 SLAM 级纯净数据 (按 Ctrl+C 退出)..." << std::endl;
    
    while (true) {
        int16_t raw_ax, raw_ay, raw_az;
        int16_t raw_gx, raw_gy, raw_gz;
        
        imu.readAccelBurst(raw_ax, raw_ay, raw_az);
        imu.readGyroBurst(raw_gx, raw_gy, raw_gz);
        
        // 转换为物理值
        double ax = raw_ax * acc_scale;
        double ay = raw_ay * acc_scale;
        double az = raw_az * acc_scale;
        
        // ⚠️ 关键动作：减去零偏误差！
        double gx = (raw_gx * gyro_scale) - bgx;
        double gy = (raw_gy * gyro_scale) - bgy;
        double gz = (raw_gz * gyro_scale) - bgz;
        
        std::printf("\r[重力 m/s2] X:%6.2f Y:%6.2f Z:%6.2f | [角速 rad/s] X:%6.3f Y:%6.3f Z:%6.3f   ", 
                    ax, ay, az, gx, gy, gz);
        std::fflush(stdout);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return 0;
}
