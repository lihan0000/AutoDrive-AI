import cv2
import os


def Extract_frames(video_path, output_dir, frames=2700, crop_size=(2160, 2160), resize_size=(1280, 1280)):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: 帧率={fps:.2f}, 总帧数={total_frames}, 尺寸={frame_W}x{frame_H}")

    # 检查视频是否合适
    if total_frames < frames:
        raise ValueError(f"视频帧数不足，需要{frames}帧，但视频只有{total_frames}帧")

    # 检查尺寸是否合适
    if crop_size[0] > frame_W or crop_size[1] > frame_H:
        raise ValueError(f"裁剪尺寸({crop_size[0]}x{crop_size[1]})超过视频原始尺寸({frame_W}x{frame_H})")

    # 计算中心裁剪区域
    start_X = (frame_W - crop_size[0]) // 2
    start_Y = (frame_H - crop_size[1]) // 2

    # 逐帧处理
    saved_count = 0

    while saved_count < frames:
        # 当前帧
        ret, frame = cap.read()
        if not ret:
            break

        # 中心裁剪
        cropped = frame[start_Y:start_Y + crop_size[1], start_X:start_X + crop_size[0]]

        # 缩放目标尺寸
        resized = cv2.resize(cropped, resize_size)

        # 保存图片
        filename = f"{saved_count:04d}.png"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized)

        saved_count += 1

        # 显示进度
        if saved_count % 100 == 0:
            print(f"已保存 {saved_count}/{frames} 张图片")

    print(f"处理完成，共保存 {saved_count} 张图片到 {output_dir}")

    # 释放资源
    cap.release()


if __name__ == "__main__":
    # 视频路径
    video_path0 = "Image/Video/Video0.mp4"

    # 输出目录
    output_dir0 = "Image/Video_frames"

    try:
        Extract_frames(video_path0, output_dir0)
        print("处理完成！")
    except Exception as e:
        print(f"处理出错: {str(e)}")
