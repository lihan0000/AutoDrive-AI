# -*- coding: utf-8 -*-
import time
import os
import spidev

# ==========================================
# 1. FastGPIO Driver
# ==========================================
class FastGPIO:
    def __init__(self, pin):
        self.pin = str(pin)
        self.path = f"/sys/class/gpio/gpio{self.pin}"
        
        if not os.path.exists(self.path):
            with open("/sys/class/gpio/export", "w") as f:
                f.write(self.pin)
            time.sleep(0.1) 
            
        with open(f"{self.path}/direction", "w") as f:
            f.write("out")
            
        self.fd = os.open(f"{self.path}/value", os.O_WRONLY)
        self.set_high()
        time.sleep(0.01)

    def set_low(self): 
        os.write(self.fd, b"0")
        
    def set_high(self): 
        os.write(self.fd, b"1")

# ==========================================
# 2. BMI088 Driver
# ==========================================
class BMI088:
    def __init__(self, gyro_pin=396):
        self.spi = spidev.SpiDev()
        self.spi.open(1, 0)
        self.spi.max_speed_hz = 5000000 
        self.spi.mode = 0
        
        self.gyro_cs = FastGPIO(gyro_pin)
        self.bgz = 0.0 
        self.gyro_scale = (1000.0 / 32768.0) 

    def write_reg(self, reg, value):
        self.gyro_cs.set_low()
        self.spi.xfer2([reg & 0x7F, value])
        self.gyro_cs.set_high()

    def initialize(self):
        print(">> Waking up BMI088 Gyroscope...")
        self.write_reg(0x15, 0x00)
        time.sleep(0.05)
        self.write_reg(0x0F, 0x01) 
        time.sleep(0.05)
        
    def read_gyro_z(self):
        self.gyro_cs.set_low()
        resp = self.spi.xfer2([0x82, 0, 0, 0, 0, 0, 0]) 
        self.gyro_cs.set_high()
        
        raw_z = (resp[6] << 8) | resp[5]
        
        if raw_z > 32767: 
            raw_z -= 65536
            
        return raw_z * self.gyro_scale

    def calibrate(self, samples=200):
        print("\nWARNING: Keep the board FLAT and STILL!")
        time.sleep(2.0)
        
        print("Calibrating zero-bias (1 sec)...")
        total_z = 0.0
        for _ in range(samples):
            total_z += self.read_gyro_z()
            time.sleep(0.005) 
            
        self.bgz = total_z / samples
        print(f"Calibration DONE! Z-Bias: {self.bgz:.4f} dps\n")

# ==========================================
# 3. Main Loop
# ==========================================
def test_imu():
    imu = BMI088()
    imu.initialize()
    imu.calibrate(samples=600)
    
    current_yaw = 0.0
    last_time = time.time()
    
    print("Starting Attitude Output (Press Ctrl+C to stop)")
    print("-" * 50)
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            gz_raw = imu.read_gyro_z()
            gz_corrected = gz_raw - imu.bgz
            
            if abs(gz_corrected) < 1.0:
                gz_corrected = 0.0 
                
            current_yaw += gz_corrected * dt
            
            direction = "RIGHT >>>" if current_yaw > 0 else "LEFT  <<<"
            if current_yaw == 0: direction = "STILL ---"
            
            bar = "=" * int(min(abs(current_yaw) / 2, 40))
            print(f"\r[YAW Angle] : {current_yaw:+7.2f} deg | {direction} {bar:<40}", end="", flush=True)
            
            time.sleep(0.005) 
            
    except KeyboardInterrupt:
        print("\n\nStopped by user.")

if __name__ == "__main__":
    test_imu()