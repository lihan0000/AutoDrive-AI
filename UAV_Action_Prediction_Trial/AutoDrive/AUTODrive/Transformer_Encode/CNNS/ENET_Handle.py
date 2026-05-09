import torch
import torch.nn as nn
from PIL import Image
import numpy as np


def create_Image_Mask_enet(Image_Mask_Path):
    class_color_ranges = \
        {
            0: ((0, 255), (0, 255), (0, 255)),    # 背景0: 初始
            1: ((100, 150), (0, 150), (0, 150)),  # 类别1: 红色
        }
    num_classes = len(class_color_ranges)
    img = Image.open(Image_Mask_Path)
    resized_img = img.resize((320, 320))
    masks = np.array(resized_img.convert('RGB'), dtype=np.uint8)
    class_masks = np.zeros((num_classes, masks.shape[0], masks.shape[1]), dtype=np.uint8)
    # 遍历每个类别
    for class_idx in class_color_ranges:
        # 获取当前类别的RGB颜色范围
        r_range, g_range, b_range = class_color_ranges[class_idx]

        # 检查每个像素是否在当前类别的颜色范围内
        # 红色通道检查
        r_mask = (masks[:, :, 0] >= r_range[0]) & (masks[:, :, 0] <= r_range[1])
        # 绿色通道检查
        g_mask = (masks[:, :, 1] >= g_range[0]) & (masks[:, :, 1] <= g_range[1])
        # 蓝色通道检查
        b_mask = (masks[:, :, 2] >= b_range[0]) & (masks[:, :, 2] <= b_range[1])

        # 合并三个通道的掩码，满足所有条件的像素为True
        class_masks[class_idx] = (r_mask & g_mask & b_mask).astype(np.uint8)
        masks_tensor = torch.from_numpy(class_masks)
        masks_tensor = masks_tensor.unsqueeze(0)

    return masks_tensor

class SimulatedHandle(nn.Module):
    def __init__(self):
        super(SimulatedHandle, self).__init__()

        self.conv = nn.Sequential(
            # 第一层卷积：64→32
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 第二层卷积：32→16
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 第三层卷积：16→8
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, Enet_O):
        # 合并批次和通道维度
        batch_size, channels, H, W = Enet_O.shape
        flattened = Enet_O.view(-1, H, W)  # [batch_size*channels, 320, 320]

        block_size = 64
        blocks_num = H // block_size  # 320//64=5

        blocks = []
        for img in flattened:
            for i in range(blocks_num):
                for j in range(blocks_num):
                    start_i = i * block_size
                    end_i = start_i + block_size
                    start_j = j * block_size
                    end_j = start_j + block_size
                    block = img[start_i:end_i, start_j:end_j]
                    blocks.append(block)

        blocks_tensor = torch.stack(blocks).unsqueeze(1)  # [N, 1, 64, 64]

        # 应用卷积层（使用模块属性中的卷积层）
        reduced_blocks = self.conv(blocks_tensor)
        reduced_blocks = reduced_blocks.squeeze(1)  # [N, 8, 8]

        # 展平并合并块
        flattened_blocks = reduced_blocks.view(reduced_blocks.size(0), -1)
        Fin_blocks = []
        for i in range(0, flattened_blocks.size(0), 2):
            combined = torch.cat([flattened_blocks[i], flattened_blocks[i + 1]], dim=0)
            Fin_blocks.append(combined)

        return torch.stack(Fin_blocks).unsqueeze(0)
