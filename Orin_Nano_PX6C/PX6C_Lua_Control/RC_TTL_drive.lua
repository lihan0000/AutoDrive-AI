local driver = CAN:get_device(5)
local port = serial:find_serial(0)

if not driver or not port then 
    gcs:send_text(0, "Hardware Init Failed!")
    return 
end

-- ================= 配置区 =================
local Z_CENTER = 400        -- 视觉偏航中心点
local KP = 15               -- 偏航P控制器系数
local MAX_RPM_AUTO = 3000   -- MAX自动驾驶转速
local ACCEL_RATE = 250      -- 调度(20ms)的转速增量(加速度)
port:begin(115200)          -- 串口波特率
-- =========================================

-- 全局状态变量 
local curr_rpm_L, target_rpm_L = 0, 0
local curr_rpm_R, target_rpm_R = 0, 0
local buffer = ""
local last_jetson_time = 0

-- 缓起缓停 (Slew Rate Limiter)
local function apply_accel(current, target)
    if current < target then
        return math.min(current + ACCEL_RATE, target)
    elseif current > target then
        return math.max(current - ACCEL_RATE, target)
    end
    return current
end

-- 构建CAN报文(协议 V5.1)
-- mode: 1正装(左侧), -1反装(右侧)
local function build_frame_v5(can_id, target_rpm, mode)
    local abs_rpm = math.min(math.floor(math.abs(target_rpm)), 10000)
    
    -- Byte3 控制逻辑: Bit4(16进制0x10)代表自动模式
    local byte3 = 0x10 

    if abs_rpm > 10 then -- 死区滤除抖动
        byte3 = byte3 | 0x01 -- Bit0: 运行使能

        -- 结合安装方向决定电机实际旋转方向
        if (target_rpm * mode) > 0 then 
            byte3 = byte3 | 0x04 -- Bit2: 前进
        else 
            byte3 = byte3 | 0x08 -- Bit3: 后退
        end
    end
    
    local msg = CANFrame()
    msg:id(can_id)
    msg:dlc(8)
    msg:data(0, abs_rpm & 0xFF)         -- 转速低位 
    msg:data(1, (abs_rpm >> 8) & 0xFF)  -- 转速高位 
    msg:data(2, byte3)                  -- 控制逻辑 
    return msg
end

function update()
    -- 确保车辆处于解锁状态
    if not arming:is_armed() then arming:arm() end

    -- 解析串口指令 (Jetson -> 飞控)
    while port:available() > 0 do
        local char_code = port:read()
        if char_code == 10 then -- '\n' ASCII码为10
            local z_val = tonumber(string.match(buffer, "Z:(%-?%d+)"))
            if z_val then
                last_jetson_time = millis()
                local error = z_val - Z_CENTER
                
                -- P控制器计算差速目标转速
                local rpm = math.min(MAX_RPM_AUTO, math.abs(error * KP))
                
                if error < 0 then 
                    -- 左转：左轮后退(负)，右轮前进(正)
                    target_rpm_L = -rpm
                    target_rpm_R = rpm
                else 
                    -- 右转：左轮前进(正)，右轮后退(负)
                    target_rpm_L = rpm
                    target_rpm_R = -rpm
                end
            end
            buffer = ""
        else
            buffer = buffer .. string.char(char_code)
            -- 防溢出保护: 单行指令过长直接丢弃
            if #buffer > 64 then buffer = "" end 
        end
    end

    -- 500ms 信号丢失安全保护(Failsafe)
    if (millis() - last_jetson_time) > 500 then
        target_rpm_L = 0
        target_rpm_R = 0
    end

    -- 应用转速爬升率 (加速度)
    curr_rpm_L = apply_accel(curr_rpm_L, target_rpm_L)
    curr_rpm_R = apply_accel(curr_rpm_R, target_rpm_R)

    -- 下发CAN指令
    if arming:is_armed() then
        -- 左侧电机 (正装 mode=1)
        driver:write_frame(build_frame_v5(0x601, curr_rpm_L, 1), 10000)
        driver:write_frame(build_frame_v5(0x621, curr_rpm_L, 1), 10000)
        -- 右侧电机 (反装 mode=-1)
        driver:write_frame(build_frame_v5(0x611, curr_rpm_R, -1), 10000)
        driver:write_frame(build_frame_v5(0x631, curr_rpm_R, -1), 10000)
    end

    return update, 20 -- 50Hz 执行频率
end

return update()
