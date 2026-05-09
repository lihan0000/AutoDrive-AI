import cv2
import os
from pathlib import Path


def specific_images_to_video(image_dir, output_video_path, fps=60):

    image_files = [f"{i:04d}.png" for i in range(2700)]

    existing_files = []
    missing_files = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        if os.path.exists(img_path):
            existing_files.append(img_file)
        else:
            missing_files.append(img_file)

    if missing_files:
        print(f"警告：共缺失 {len(missing_files)} 个文件")
        if len(missing_files) < 10:  # 只显示前10个缺失文件
            print("缺失的文件：", ", ".join(missing_files[:10]))

    if not existing_files:
        print("错误：未找到任何符合要求的图片文件")
        return

    first_image_path = os.path.join(image_dir, existing_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"错误：无法读取图片 {first_image_path}")
        return

    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_images = len(image_files)
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)

        if not os.path.exists(image_path):
            print(f"警告：跳过缺失的图片 {image_path}")
            continue

        frame = cv2.imread(image_path)

        if frame is None:
            print(f"警告：无法读取图片 {image_path}，已跳过")
            continue

        # 确保所有帧尺寸一致
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
            print(f"警告：图片 {image_file} 尺寸不一致，已调整")

        video.write(frame)

        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{total_images} 张图片")

    # 释放资源
    video.release()
    cv2.destroyAllWindows()
    print(f"视频合成完成！保存路径：{output_video_path}")
    print(f"处理结果：成功 {len(existing_files)} 个，缺失 {len(missing_files)} 个")


if __name__ == "__main__":
    # 配置路径
    input_image_dir = "Image/Video/frames_command"  # 替换为你的图片文件夹路径
    output_video_path = "I_video.mp4"  # 输出视频文件名

    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 调用函数合成视频，帧率设为60
    specific_images_to_video(input_image_dir, output_video_path, fps=60)
