##########################################################################################################
import torch  # Torch核心库，用于张量操作和神经网络
import torch.nn as nn  # 神经网络模块，包含各种层和损失函数
import torch.optim as optim  # 优化器模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
from torchvision import transforms  # 图像预处理工具
import numpy as np  # 数值计算库
from PIL import Image
import os  # 文件操作库
import random
from tqdm import tqdm  # 进度条显示工具
from model import Enet  # 导入自定义的Enet模型


def Train_image_transform():
    return transforms.Compose([
        transforms.CenterCrop(size=(320, 320)),
        # 图像与掩码同步
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转：50%概率
        transforms.RandomVerticalFlip(p=0.2),    # 垂直翻转：20%概率
        transforms.RandomAffine
        (
            degrees=(-15, 15),                                    # 随机旋转角度范围：-15到15
            scale=(0.8, 1.2),                                     # 随机缩放范围：80%到120%
            shear=(-10, 10, -10, 10),                             # 剪切范围：X轴-10到10，Y轴-10到10
            translate=(0.1, 0.1),                                 # 平移范围：宽度和高度的10%以内
            fill=(0, 0, 0),                                       # 填充黑色：注意通道数
            interpolation=transforms.InterpolationMode.BILINEAR,  # 图像插值
        ),

        # 可以增加局部形变（弹性形变）

        # 单图像
        transforms.ColorJitter(
            brightness=0.15,                                      # 亮度±15%（弱于常规增强）
            contrast=0.15,                                        # 对比度±15%
            saturation=0.15,                                      # 饱和度±15%
            hue=0.05                                              # 色相±5%（极小范围，避免色偏严重）
        ),

        transforms.RandomGrayscale(p=0.05),                             # 低概率灰度化：模拟色彩丢失
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),  # 轻度锐化调整：增强边缘特征

        # 可以增加画质模拟（高斯噪声）

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225])

    ])

def Train_masks_transform():
    return transforms.Compose([
        transforms.CenterCrop(size=(320, 320)),
        # 图像与掩码同步
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转：50%概率
        transforms.RandomVerticalFlip(p=0.2),    # 垂直翻转：20%概率
        transforms.RandomAffine
        (
            degrees=(-15, 15),                                    # 随机旋转角度范围：-15到15
            scale=(0.8, 1.2),                                     # 随机缩放范围：80%到120%
            shear=(-10, 10, -10, 10),                             # 剪切范围：X轴-10到10，Y轴-10到10
            translate=(0.1, 0.1),                                 # 平移范围：宽度和高度的10%以内
            fill=0,                                               # 填充黑色：注意通道数
            interpolation=transforms.InterpolationMode.NEAREST,   # 图像插值
        ),

        # 可以增加局部形变（弹性形变）

        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long().squeeze(0))

    ])

def Val_image_transform():
    return transforms.Compose([
        transforms.CenterCrop(size=(320, 320)),  # 剪切为适合模型的尺寸
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225])
    ])

# 定义数据集类，继承自PyTorch的Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, masks_dir, transform=True):
        # 初始化数据集
        self.image_dir = image_dir  # 图像文件夹路径
        self.masks_dir = masks_dir  # 掩码文件夹路径
        self.transform = transform  # 图像预处理变换
        self.images = os.listdir(image_dir)  # 获取所有图像文件名

        # 定义颜色分类
        self.class_color_ranges =\
        {
            0: ((0, 255), (0, 255), (0, 255)),        # 背景0: 初始
            1: ((100, 150), (0, 150), (0, 150)),      # 类别1: 红色
        }
        self.num_classes = len(self.class_color_ranges)

    def __len__(self):
        # 返回数据集大小
        return len(self.images)

    def __getitem__(self, idx):
        # 根据索引获取数据样本
        image_path = os.path.join(self.image_dir, self.images[idx])  # 图像路径
        masks_path = os.path.join(self.masks_dir, self.images[idx])  # 掩码路径

        # 读取图像和掩码（支持npy格式和普通图像格式）
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)  # 掩码为三通道

        if masks_path.endswith('.npy'):
            masks = np.load(masks_path)
        else:
            masks = np.array(Image.open(masks_path).convert('RGB'), dtype=np.uint8)  # 掩码为三通道
            class_masks = np.zeros((masks.shape[0], masks.shape[1]), dtype=np.uint8)
            for class_idx, (r_range, g_range, b_range) in self.class_color_ranges.items():
                r_match = (masks[..., 0] >= r_range[0]) & (masks[..., 0] <= r_range[1])
                g_match = (masks[..., 1] >= g_range[0]) & (masks[..., 1] <= g_range[1])
                b_match = (masks[..., 2] >= b_range[0]) & (masks[..., 2] <= b_range[1])
                matches = r_match & g_match & b_match
                class_masks[matches] = class_idx
            masks = class_masks


        # 应用预处理变换
        if self.transform:
            seed = random.randint(0, 2 ** 32 - 1)
            image = Image.fromarray(image)
            masks = Image.fromarray(masks)
            image_transform = Train_image_transform()
            masks_transform = Train_masks_transform()
            random.seed(seed)
            torch.manual_seed(seed)
            image = image_transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            masks = masks_transform(masks)

        return image, masks

def train_Enet(image_path, masks_path, epochs=5, save_path="saved_models"):
    dataset = SegmentationDataset(image_dir=image_path, masks_dir=masks_path)  # 数据集生成
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)              # 数据加载器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                    # 训练
    model = Enet(num_classes=dataset.num_classes).to(device)                   # 模型
    criterion = nn.CrossEntropyLoss()                                          # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)                        # 优化函数

    # 记录最佳损失，用于保存最佳模型
    best_loss = float('inf')

    for epoch in tqdm(range(epochs), desc="进度", unit="轮"):
        model.train()
        epoch_loss = 0.0

        for images, masks in tqdm(dataloader, desc=f"第{epoch + 1}轮", unit="批", leave=False):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            OUTPUTS = model(images)
            loss = criterion(OUTPUTS, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # 累加所有批次的损失

        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        tqdm.write(f"第{epoch + 1}轮 - 平均损失: {avg_epoch_loss:.6f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        tqdm.write(f"第{epoch + 1}轮 - 平均损失: {avg_epoch_loss:.6f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_Path = os.path.join(save_path, "enet_best.pth")
            torch.save(model.state_dict(), best_model_Path)
            tqdm.write(f"模型已保存至: {best_model_Path}")

    return model

def Predict_and_save_masks(image_dir, model_path, class_colors, save_dir="Predicted_masks"):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(class_colors)
    model = Enet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    image_files = [f for f in os.listdir(image_dir) if
                   (f.endswith(('.png', '.npy')) and
                    not f.startswith('.'))]
    transform = Val_image_transform()
    with torch.no_grad():
        for img_file in tqdm(image_files, desc="预测掩码", unit="张"):
            img_path = os.path.join(image_dir, img_file)
            if img_path.endswith('.npy'):
                image = np.load(img_path)
            else:
                image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            original_size = image.shape[:2]
            img_pil = Image.fromarray(image)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            OUTPUT = model(img_tensor)
            Pred_mask = torch.argmax(OUTPUT, dim=1).squeeze().cpu().numpy()

            color_mask = np.zeros((Pred_mask.shape[0], Pred_mask.shape[1], 3), dtype=np.uint8)

            for class_idx, color in class_colors.items():
                color_mask[Pred_mask == class_idx] = color

            # 将numpy数组转换为PIL图像，便于调整尺寸
            color_mask_pil = Image.fromarray(color_mask)

            # 将掩码调整回原始图像尺寸（注意宽高顺序）
            color_mask_resized = color_mask_pil.resize(
                (original_size[1], original_size[0]),
                Image.Resampling.NEAREST
            )

            save_name = os.path.splitext(img_file)[0] + "_Pred_mask.png"

            save_path = os.path.join(save_dir, save_name)

            color_mask_resized.save(save_path)

        tqdm.write(f"所有预测掩码已保存至: {save_dir}")


if __name__ == "__main__":
    # 单张图片和掩码路径
    IMAGE_PATH = "Image"  # 训练/验证图片路径
    MASKS_PATH = "Masks"  # 掩码路径
    SAVE_PATH = "saved_models"
    PREDICTED_MASK_PATH = "Predicted_masks"

    # 训练模型（只接收模型返回值）
    model = train_Enet(
        image_path=IMAGE_PATH,
        masks_path=MASKS_PATH,
        epochs=50,
        save_path=SAVE_PATH
    )

    class_colors = {
        0: (0,   255, 0),
        1: (128, 0, 0)
    }

    # 加载最佳模型并预测
    best_model_path = os.path.join(SAVE_PATH, "enet_best.pth")
    if os.path.exists(best_model_path):
        Predict_and_save_masks(
            image_dir=IMAGE_PATH,
            model_path=best_model_path,
            class_colors=class_colors,
            save_dir=PREDICTED_MASK_PATH
        )
    else:
        print(f"未找到模型文件: {best_model_path}")
##########################################################################################################