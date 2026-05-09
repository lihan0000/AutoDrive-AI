-- ==============================================================================
-- Script Name: agv_motor_control.lua (Production Release)
-- Protocol: VCU to Motor Controller V5.1 (4WD)
-- Description: CAN bus control, telemetry, and odometry integration for ArduRover.
-- ==============================================================================

local MAX_RPM = 3000.0
local CAN_DRV = CAN:get_device(25)

-- Physical Configuration
local DIR_L, DIR_R = 1, -1
local TX_LF, TX_LR = 0x601, 0x621
local TX_RF, TX_RR = 0x611, 0x631
local RX_LF, RX_LR = 0x603, 0x623
local RX_RF, RX_RR = 0x613, 0x633

-- State Tracking Variables
local last_time_ms = millis()
local last_print_ms = 0
local last_fault_ms = {}
local phase_L, phase_R = 0.0, 0.0
local boot_time_ms = millis()

-- Telemetry Variables
local act_rpm_L, act_rpm_R = 0.0, 0.0

-- Utility: Safe Number Conversion
local function to_num(val)
    if val == nil then return 0 end
    if type(val) == 'number' then return val end
    if type(val) == 'userdata' and val.toint then return val:toint() end
    return tonumber(val) or 0
end

-- Utility: Value Clamping
local function clamp(val, min_val, max_val)
    return math.max(math.min(val, max_val), min_val)
end

-- ==============================================================================
-- TX: Generate Control Commands
-- ==============================================================================
local function get_ctrl_byte(target_val, is_armed)
    local is_run = 1
    local is_auto = 1
    local is_forward, is_reverse = 0, 0

    if is_armed and math.abs(target_val) > 10 then
        if target_val > 0 then is_forward = 1 else is_reverse = 1 end
    end

    return (is_run << 0) | (0 << 1) | (is_forward << 2) | (is_reverse << 3) | (is_auto << 4)
end

local function send_tx(id, target_val, is_armed)
    if not CAN_DRV then return end
    local safe_rpm = math.floor(math.abs(clamp(target_val, -MAX_RPM, MAX_RPM)))

    local msg = CANFrame()
    msg:id(id)
    msg:dlc(8)

    msg:data(0, safe_rpm & 0xFF)
    msg:data(1, (safe_rpm >> 8) & 0xFF)
    msg:data(2, get_ctrl_byte(target_val, is_armed))
    for i = 3, 7 do msg:data(i, 0) end

    CAN_DRV:write_frame(msg, 10000)
end

-- ==============================================================================
-- RX: Parse Feedback & Odometry
-- ==============================================================================
local function parse_motor_feedback(frame, dir_multiplier, dt_s)
    local abs_rpm = (to_num(frame:data(1)) & 0xFF) | ((to_num(frame:data(2)) & 0xFF) << 8)
    local dir_raw = to_num(frame:data(0))
    local dir_bits = (dir_raw >> 4) & 0x03

    local sign = 0
    if dir_bits == 1 then sign = 1 elseif dir_bits == 2 then sign = -1 end

    local current_rpm = abs_rpm * sign * dir_multiplier
    local delta_radians = (current_rpm / 60.0) * math.pi * 2.0 * dt_s

    return current_rpm, delta_radians
end

local function process_feedback()
    if not CAN_DRV then return end

    local frame = CAN_DRV:read_frame()
    if not frame then return end

    local now_ms = millis()
    local dt_s = (now_ms - last_time_ms) * 0.001
    if dt_s <= 0 then dt_s = 0.001 end
    local dt_us = math.floor(dt_s * 1000000)
    last_time_ms = now_ms

    local d_LF, d_LR, d_RF, d_RR = nil, nil, nil, nil

    while frame do
        local id = to_num(frame:id())
        local fault = to_num(frame:data(3))

        if fault > 0 and (now_ms - (last_fault_ms[id] or 0)) > 2000 then
            gcs:send_text(1, string.format("Motor 0x%X Fault: %d", id, fault))
            last_fault_ms[id] = now_ms
        end

        if id == RX_LF then act_rpm_L, d_LF = parse_motor_feedback(frame, DIR_L, dt_s)
        elseif id == RX_LR then _, d_LR = parse_motor_feedback(frame, DIR_L, dt_s)
        elseif id == RX_RF then act_rpm_R, d_RF = parse_motor_feedback(frame, DIR_R, dt_s)
        elseif id == RX_RR then _, d_RR = parse_motor_feedback(frame, DIR_R, dt_s)
        end

        frame = CAN_DRV:read_frame()
    end

    if wheel_encoder then
        local final_L = d_LF or d_LR
        local final_R = d_RF or d_RR

        if final_L then
            phase_L = phase_L + final_L
            wheel_encoder:update(0, phase_L, 100, dt_us)
        end
        if final_R then
            phase_R = phase_R + final_R
            wheel_encoder:update(1, phase_R, 100, dt_us)
        end
    end
end

-- ==============================================================================
-- Main Control Loop (20Hz)
-- ==============================================================================
function update()
    if not CAN_DRV then return update, 1000 end

    process_feedback()

    local now = millis()
    local is_init = (now - boot_time_ms < 2000)
    local armed = arming:is_armed()
    local target_L, target_R = 0.0, 0.0

    if not is_init and armed then
        target_L = to_num(SRV_Channels:get_output_scaled(73)) * 0.001 * MAX_RPM * DIR_L
        target_R = to_num(SRV_Channels:get_output_scaled(74)) * 0.001 * MAX_RPM * DIR_R

        if (now - last_print_ms) > 1000 then
            local odo_L, odo_R = 0.0, 0.0
            if wheel_encoder then
                odo_L = wheel_encoder:get_distance(0) or 0.0
                odo_R = wheel_encoder:get_distance(1) or 0.0
            end
            gcs:send_text(6, string.format("L:T%.0f A%.0f O%.2f | R:T%.0f A%.0f O%.2f",
                target_L, act_rpm_L, odo_L, target_R, act_rpm_R, odo_R))
            last_print_ms = now
        end
    end

    send_tx(TX_LF, target_L, armed)
    send_tx(TX_LR, target_L, armed)
    send_tx(TX_RF, target_R, armed)
    send_tx(TX_RR, target_R, armed)

    return update, 50
end

return update, 1000