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
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


def Train_masks_transform():
    return transforms.Compose([
        transforms.CenterCrop(size=(320, 320)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long().squeeze(0))  # 转换为类别索引
    ])


def Val_image_transform():
    return transforms.Compose([
        transforms.CenterCrop(size=(320, 320)),  # 剪切为适合模型的尺寸
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


# 定义数据集类，继承自PyTorch的Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, masks_dir, transform=True):
        # 初始化数据集
        self.image_dir = image_dir  # 图像文件夹路径
        self.masks_dir = masks_dir  # 掩码文件夹路径
        self.transform = transform  # 图像预处理变换

        # 获取所有图像文件名（排除隐藏文件）
        self.images = [f for f in os.listdir(image_dir)
                       if not f.startswith('.') and
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))]

        # 定义颜色分类：背景(0)为纯黑色，A类(1)为红色(255,0,0)，B类(2)为绿色(0,255,0)
        self.class_color_ranges = {
            0: ((0, 0), (0, 0), (0, 0)),  # 背景：纯黑色
            1: ((255, 255), (0, 0), (0, 0)),  # A类：红色 (255,0,0)
            2: ((0, 0), (255, 255), (0, 0))  # B类：绿色 (0,255,0)
        }
        self.num_classes = len(self.class_color_ranges)

    def __len__(self):
        # 返回数据集大小
        return len(self.images)

    def __getitem__(self, idx):
        # 根据索引获取数据样本
        img_filename = self.images[idx]
        image_path = os.path.join(self.image_dir, img_filename)  # 图像路径

        # 掩码文件名为图像文件名基础上添加"_mask"后缀（核心修改点）
        img_basename = os.path.splitext(img_filename)[0]
        img_ext = os.path.splitext(img_filename)[1]
        mask_basename = f"{img_basename}_mask"  # 生成掩码文件的基础名称（如"1" -> "1_mask"）
        mask_extensions = ['.png', '.jpg', '.jpeg', '.npy']
        masks_path = None

        # 查找对应掩码文件
        for ext in mask_extensions:
            candidate = os.path.join(self.masks_dir, f"{mask_basename}{ext}")
            if os.path.exists(candidate):
                masks_path = candidate
                break

        if masks_path is None:
            raise FileNotFoundError(f"未找到 {img_filename} 对应的掩码文件（预期掩码名：{mask_basename}.*）")

        # 读取图像
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)

        # 读取掩码
        if masks_path.endswith('.npy'):
            masks = np.load(masks_path)
        else:
            masks = np.array(Image.open(masks_path).convert('RGB'), dtype=np.uint8)
            class_masks = np.zeros((masks.shape[0], masks.shape[1]), dtype=np.uint8)

            # 根据颜色范围生成类别掩码
            for class_idx, (r_range, g_range, b_range) in self.class_color_ranges.items():
                r_match = (masks[..., 0] >= r_range[0]) & (masks[..., 0] <= r_range[1])
                g_match = (masks[..., 1] >= g_range[0]) & (masks[..., 1] <= g_range[1])
                b_match = (masks[..., 2] >= b_range[0]) & (masks[..., 2] <= b_range[1])
                matches = r_match & g_match & b_match
                class_masks[matches] = class_idx
            masks = class_masks

        # 应用预处理变换（保持图像和掩码变换同步）
        if self.transform:
            seed = random.randint(0, 2 **32 - 1)
            image = Image.fromarray(image)
            masks = Image.fromarray(masks)

            image_transform = Train_image_transform()
            masks_transform = Train_masks_transform()

            # 确保图像和掩码使用相同的随机种子
            random.seed(seed)
            torch.manual_seed(seed)
            image = image_transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            masks = masks_transform(masks)

        return image, masks


def train_Enet(image_dir, masks_dir, epochs=5, save_path="saved_models"):
    # 创建保存模型的目录
    os.makedirs(save_path, exist_ok=True)

    # 加载数据集和数据加载器
    dataset = SegmentationDataset(image_dir=image_dir, masks_dir=masks_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)  # 增大batch_size并启用多线程

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 初始化模型、损失函数和优化器
    model = Enet(num_classes=dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 记录最佳损失，用于保存最佳模型
    best_loss = float('inf')

    for epoch in tqdm(range(epochs), desc="训练进度", unit="轮"):
        model.train()
        epoch_loss = 0.0

        for images, masks in tqdm(dataloader, desc=f"第{epoch + 1}轮", unit="批", leave=False):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算并显示平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        tqdm.write(f"第{epoch + 1}轮 - 平均损失: {avg_epoch_loss:.6f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(save_path, "enet_best.pth")
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"最佳模型已保存至: {best_model_path}")

    return model


def Predict_and_save_masks(input_image_dir, model_path, class_colors, save_dir="Predicted_masks"):
    # 创建保存预测结果的目录
    os.makedirs(save_dir, exist_ok=True)

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(class_colors)

    # 加载模型
    model = Enet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 获取所有待预测图像（排除隐藏文件）
    image_files = [f for f in os.listdir(input_image_dir)
                   if not f.startswith('.') and
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))]

    transform = Val_image_transform()

    with torch.no_grad():  # 关闭梯度计算，加速预测
        for img_file in tqdm(image_files, desc="预测进度", unit="张"):
            img_path = os.path.join(input_image_dir, img_file)

            # 读取图像
            if img_path.endswith('.npy'):
                image = np.load(img_path)
            else:
                image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

            original_size = image.shape[:2]  # 保存原始尺寸用于恢复
            img_pil = Image.fromarray(image)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)  # 增加批次维度

            # 模型预测
            outputs = model(img_tensor)
            pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()  # 获取类别索引

            # 将预测结果转换为彩色掩码
            color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            for class_idx, color in class_colors.items():
                color_mask[pred_mask == class_idx] = color

            # 将掩码调整回原始图像尺寸
            color_mask_pil = Image.fromarray(color_mask)
            color_mask_resized = color_mask_pil.resize(
                (original_size[1], original_size[0]),  # (width, height)
                Image.Resampling.NEAREST
            )

            # 保存预测结果
            save_name = os.path.splitext(img_file)[0] + "_Pred_mask.png"
            save_path = os.path.join(save_dir, save_name)
            color_mask_resized.save(save_path)

        tqdm.write(f"所有预测掩码已保存至: {save_dir}")


if __name__ == "__main__":
    # 路径配置 - 可根据需要修改
    IMAGE_DIR = "Image/image"  # 训练图像文件夹
    MASKS_DIR = "Masks/mask"  # 掩码文件夹
    SAVE_PATH = "saved_models"  # 模型保存目录
    PREDICT_INPUT_DIR = "Image_ALL/Video_image"  # 预测输入图像文件夹（可修改）
    PREDICT_OUTPUT_DIR = "Masks_ALL/Video_masks"  # 预测结果保存目录
    """
    # 训练模型
    model = train_Enet(
        image_dir=IMAGE_DIR,
        masks_dir=MASKS_DIR,
        epochs=500,
        save_path=SAVE_PATH
    )
    """
    # 定义预测时的类别颜色（与训练时对应）
    class_colors = {
        0: (0, 0, 0),  # 背景：纯黑色
        1: (255, 0, 0),  # A类：红色
        2: (0, 255, 0)  # B类：绿色
    }

    # 加载最佳模型并预测
    best_model_path = os.path.join(SAVE_PATH, "enet_best.pth")
    if os.path.exists(best_model_path):
        Predict_and_save_masks(
            input_image_dir=PREDICT_INPUT_DIR,  # 预测输入文件夹，可修改
            model_path=best_model_path,
            class_colors=class_colors,
            save_dir=PREDICT_OUTPUT_DIR
        )
    else:
        print(f"未找到模型文件: {best_model_path}")

##########################################################################################################