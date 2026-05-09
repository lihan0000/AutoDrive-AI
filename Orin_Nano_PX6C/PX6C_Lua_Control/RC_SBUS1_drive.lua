--[[
================================================================================
    差速小车底盘 CAN 直控脚本 (存档版)
================================================================================
    功能描述: 
        通过读取遥控器通道 (CH1:转向, CH2:油门)，使用 CAN 总线直接控制差速底盘。
        内置平滑加速算法与 RPM 软限幅，确保底盘运行平稳并保护硬件。
    
    核心特性:
        - 独立的最大转速限制 (默认 6000 RPM)
        - 渐进式加速限制算法 (防止突发大电流和机械冲击)
        - 遥控器中位死区过滤
        - 左右侧电机独立控制 (内置右侧镜像反转逻辑)
================================================================================
]]--

local can_driver = CAN:get_device(5)
if not can_driver then 
    gcs:send_text(4, "CAN: 未找到设备 5!")
    return 
end

-- ============================================================================
-- 配置参数区 (Configuration)
-- ============================================================================
local MAX_RPM    = 6000   -- 最大转速限制 (RPM)
local ACCEL_RATE = 200    -- 渐变速率：每 20ms 允许的最大 RPM 变化量 (数值越小越平滑)
local DEADZONE   = 40     -- 遥控器中位死区范围 (1500 +/- 40)

-- ============================================================================
-- 状态变量区 (State Variables)
-- ============================================================================
local current_rpm_left  = 0
local current_rpm_right = 0
local debug_log_counter = 0

--[[
    平滑增量算法 (Slew Rate Limiter)
    @param current : 当前的 RPM 值
    @param target  : 期望达到的目标 RPM 值
    @return        : 经过平滑处理后的新 RPM 值
]]
local function calculate_smooth_rpm(current, target)
    local diff = target - current
    if diff > ACCEL_RATE then
        return current + ACCEL_RATE
    elseif diff < -ACCEL_RATE then
        return current - ACCEL_RATE
    else
        return target
    end
end

--[[
    CAN 报文生成器 (Motor Frame Builder)
    @param can_id     : 目标电机驱动器的 CAN ID
    @param target_rpm : 经过计算后的目标 RPM
    @param mode       : 1 为左侧标准逻辑，-1 为右侧镜像反向逻辑
    @return           : 构造好的 CANFrame 对象
]]
local function build_motor_can_frame(can_id, target_rpm, mode)
    local abs_rpm = math.floor(math.abs(target_rpm))
    
    -- 硬件保护：硬性上限拦截
    if abs_rpm > 10000 then 
        abs_rpm = 10000 
    end 
    
    local is_running = 0
    local is_forward = 0
    local is_reverse = 0
    
    if abs_rpm > 0 then
        is_running = 1
        -- 根据物理安装模式 (mode) 决定实际的正反转指令
        local logical_direction = target_rpm * mode
        if logical_direction > 0 then 
            is_forward = 1 
        elseif logical_direction < 0 then 
            is_reverse = 1 
        end
    end
    
    -- 控制字节组装：bit0=运行, bit2=正转, bit3=反转, bit4=使能(固定为1)
    local control_byte = (is_running << 0) | (is_forward << 2) | (is_reverse << 3) | (1 << 4)
    
    local msg = CANFrame()
    msg:id(can_id)
    msg:dlc(8)
    msg:data(0, abs_rpm & 0xFF)
    msg:data(1, (abs_rpm >> 8) & 0xFF)
    msg:data(2, control_byte)
    
    return msg
end

--[[
    主控制循环 (Main Loop)
]]
function update()
    -- 1. 强制解锁检查
    if not arming:is_armed() then 
        arming:arm() 
    end

    -- 2. 读取遥控器原始信号 (CH1:转向, CH2:油门)
    local rc_steer_pwm    = rc:get_pwm(1)
    local rc_throttle_pwm = rc:get_pwm(2)
    
    -- 3. 信号预处理与死区过滤
    local steer_offset    = rc_steer_pwm - 1500
    local throttle_offset = rc_throttle_pwm - 1500
    
    if math.abs(steer_offset) < DEADZONE then 
        steer_offset = 0 
    end
    if math.abs(throttle_offset) < DEADZONE then 
        throttle_offset = 0 
    end

    -- 4. 差速混控逻辑 (将 PWM 偏移量映射到 -1.0 至 1.0 的百分比)
    local target_pct_left  = (throttle_offset + steer_offset) / 500.0
    local target_pct_right = (throttle_offset - steer_offset) / 500.0
    
    -- 5. 计算目标转速 (并应用 MAX_RPM 限幅)
    local target_rpm_left  = math.max(-MAX_RPM, math.min(MAX_RPM, target_pct_left  * MAX_RPM))
    local target_rpm_right = math.max(-MAX_RPM, math.min(MAX_RPM, target_pct_right * MAX_RPM))

    -- 6. 应用渐变加速平滑处理
    current_rpm_left  = calculate_smooth_rpm(current_rpm_left,   target_rpm_left)
    current_rpm_right = calculate_smooth_rpm(current_rpm_right, target_rpm_right)

    -- 7. 派发 CAN 报文
    if arming:is_armed() then
        local timeout_us = 10000
        
        -- 左侧：标准方向逻辑 (mode = 1)
        can_driver:write_frame(build_motor_can_frame(0x601, current_rpm_left,  1), timeout_us)
        can_driver:write_frame(build_motor_can_frame(0x621, current_rpm_left,  1), timeout_us)
        
        -- 右侧：镜像反向逻辑 (mode = -1)
        can_driver:write_frame(build_motor_can_frame(0x611, current_rpm_right, -1), timeout_us)
        can_driver:write_frame(build_motor_can_frame(0x631, current_rpm_right, -1), timeout_us)
    end

    -- 8. 地面站调试信息打印 (50Hz 循环下，每 50 帧打印一次 = 1Hz)
    debug_log_counter = debug_log_counter + 1
    if debug_log_counter >= 50 then
        gcs:send_text(6, string.format("RPM_L: %d | RPM_R: %d", current_rpm_left, current_rpm_right))
        debug_log_counter = 0
    end

    -- 重新调度: 20ms 后再次执行 (50Hz 运行频率)
    return update, 20 
end

-- 启动脚本
return update()
