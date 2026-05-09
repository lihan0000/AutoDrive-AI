import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# 配置路径
excel_path = "Command/Auto.xlsx"
input_img_dir  =   "Image/Video/Video_frames"
output_img_dir = "Image/Video/frames_command"

# 创建输出文件夹（如果不存在）
Path(output_img_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_excel(excel_path, sheet_name=0, header=None)
total_rows = min(2700, len(df))

# 配置显示参数
text_color = (255, 0, 0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
font_scale = 0.8
thickness = 1
line_type = cv2.LINE_AA

# 逐行处理数据并绘制到对应图片
for i in range(total_rows):
    # 读取当前行控制参数
    controls = df.iloc[i, 0:8].astype(float).values
    f_acc, b_acc, l_acc, r_acc, u_acc, d_acc, lc_acc, rc_acc = controls

    # 构建图片路径
    img_name = f"{i:04d}.png"
    input_img_path = os.path.join(input_img_dir, img_name)

    # 读取图片，如果不存在则跳过
    img = cv2.imread(input_img_path)
    if img is None:
        print(f"警告：未找到图片 {input_img_path}，已跳过")
        continue

    # 在图片上绘制文本
    cv2.putText(img, f"F_ACC{f_acc:.2f}", (5, 80),
                font, font_scale, text_color, thickness, line_type)
    cv2.putText(img, f"B_ACC{b_acc:.2f}", (5, 240),
                font, font_scale, text_color, thickness, line_type)
    cv2.putText(img, f"L_ACC{l_acc:.2f}", (5, 400),
                font, font_scale, text_color, thickness, line_type)
    cv2.putText(img, f"R_ACC{r_acc:.2f}", (5, 560),
                font, font_scale, text_color, thickness, line_type)
    cv2.putText(img, f"U_ACC{u_acc:.2f}", (5, 720),
                font, font_scale, text_color, thickness, line_type)
    cv2.putText(img, f"D_ACC{d_acc:.2f}", (5, 880),
                font, font_scale, text_color, thickness, line_type)
    cv2.putText(img, f"LC_ACC{lc_acc:.2f}", (5, 1100),
                font, font_scale, text_color, thickness, line_type)
    cv2.putText(img, f"RC_ACC{rc_acc:.2f}", (5, 1250),
                font, font_scale, text_color, thickness, line_type)

    # 保存处理后的图片
    output_img_path = os.path.join(output_img_dir, img_name)
    cv2.imwrite(output_img_path, img)

    # 打印进度信息
    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1}/{total_rows} 张图片")

print("所有图片处理完成！")