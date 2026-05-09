import json
import cv2
import numpy as np
import os


def mask_json(imgPath, jsonPath, Out_Path, class_colors=None):
    # 定义类别颜色映射
    class_colors = {
        "Grass": (0, 0, 255),
        "Tree" : (0, 255, 0),
    }

    with open(jsonPath, 'r') as f:
        ret_dic = json.load(f)

    img = cv2.imread(imgPath)
    if img is None:
        print(f"警告：无法读取图片 {imgPath}")
        return None

    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)  # 创建彩色掩码

    # 绘制每个形状区域
    for shape in ret_dic.get("shapes", []):
        label = shape["label"]
        color = class_colors.get(label, (255, 255, 255))  # 默认为白色
        points = np.array(shape["points"], dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], color)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(Out_Path), exist_ok=True)
    # 保存掩码图片
    cv2.imwrite(Out_Path, mask)
    return mask


def process_batch(image_dir, json_dir, output_dir, start=1, end=90):
    for i in range(start, end + 1):
        # 构建文件路径
        img_path = os.path.join(image_dir, f"{i}.png" )
        json_path = os.path.join(json_dir, f"{i}.json")
        out_path = os.path.join(output_dir, f"{i}_mask.png")

        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"跳过：PNG 文件不存在 {img_path}" )
            continue
        if not os.path.exists(json_path):
            print(f"跳过：JSON文件不存在 {json_path}")
            continue

        # 生成掩码
        mask = mask_json(img_path, json_path, out_path)
        if mask is not None:
            print(f"已处理：{i}/{end}，保存至 {out_path}")


# 配置路径
image_directory = "Image/Dateset/Train_frames"  # PNG 文件夹路径
json_directory  = "Image/Dateset/Train_mask"    # JSON文件夹路径
output_directory = "Image/Dateset/Train_label"  # 掩码文件夹路径

process_batch(image_directory, json_directory, output_directory, 1, 90)

print("批量处理完成！")

