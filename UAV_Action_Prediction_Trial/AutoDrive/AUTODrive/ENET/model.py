import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class EnetInitialBlock(nn.Module):
    """Enet初始化模块

    该模块通过并行的卷积和池化操作对输入图像进行首次下采样，
    同时融合两种操作的特征并进行批归一化和激活处理。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        """初始化Enet初始化模块

        参数:
            in_channels  : 输入图像的通道数，默认3（RGB图像）
            out_channels : 输出特征图的通道数，默认16（需大于输入通道数）

        异常:
            ValueError: 当输出通道数小于等于输入通道数时抛出
        """
        super().__init__()

        # 验证输出通道数有效性
        if out_channels <= in_channels:
            raise ValueError(
                f"输出通道数必须大于输入通道数，当前输出通道数: {out_channels}, 输入通道数: {in_channels}"
            )

        # 主分支：3x3卷积进行下采样（步长2）
        self.reduce_conv = nn.Conv2d(
            in_channels =in_channels ,
            out_channels=out_channels - in_channels,  # 减去池化分支的通道数
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False  # 后续有BN层，无需偏置
        )
        self.bn_reduce = nn.BatchNorm2d(out_channels - in_channels)

        # 跳跃分支：2x2最大池化进行下采样（步长2）
        self.skip_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 融合后的批归一化和激活函数
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.P_relu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels,  height  ,  width   )

        返回:
            x: 输出张量，形状为 (batch_size, out_channels, height//2, width//2)
        """
        # 主分支卷积操作
        main = self.reduce_conv(x)
        main = self.bn_reduce(main)

        # 跳跃分支池化操作
        skip = self.skip_pool(x)

        # 通道维度融合
        x = torch.cat([main, skip], dim=1)

        # 最终处理
        x = self.bn_final(x)
        x = self.P_relu(x)

        return x


class EnetDownsampleBottleneck(nn.Module):
    """Enet下采样瓶颈模块

    通过1x1卷积降维、3x3卷积下采样、1x1卷积升维 序列操作，
    结合残差连接实现特征图的下采样（尺寸减半）和通道数调整。
    """

    def __init__(self, in_channels: int = 16, out_channels: int = 64, reduction_ratio: int = 4, dropout_prob: float = 0.01) -> None:
        """初始化Enet下采样瓶颈模块

        参数:
            in_channels:     输入特征图的通道数
            out_channels:    输出特征图的通道数
            reduction_ratio: 降维倍率，中间通道数 = in_channels // reduction_ratio
            dropout_prob:    Dropout概率，默认0.01（适合第一阶段）

        异常:
            ValueError: 当降维倍率导致中间通道数为0时抛出
        """
        super().__init__()

        # 计算并验证中间通道数
        mid_channels = in_channels // reduction_ratio
        if mid_channels <= 0:
            raise ValueError(
                f"降维倍率({reduction_ratio})过大，导致中间通道数为0，输入通道数: {in_channels}"
            )

        # 主分支：降维 -> 卷积下采样 -> 升维
        self.reduce_conv = nn.Conv2d(
            in_channels =in_channels ,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(mid_channels)

        self.main_conv = nn.Conv2d(
            in_channels =mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,  # 下采样（尺寸减半）
            padding=1,
            bias=False
        )
        self.bn_main = nn.BatchNorm2d(mid_channels)

        self.expand_conv = nn.Conv2d(
            in_channels =mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_expand = nn.BatchNorm2d(out_channels)

        # 跳跃分支：池化 + 通道调整
        self.skip_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_conv = nn.Conv2d(
            in_channels =in_channels ,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_skip = nn.BatchNorm2d(out_channels)

        # 激活函数和Dropout
        self.P_relu = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels,  height  ,  width   )

        返回:
            x: 输出张量，形状为 (batch_size, out_channels, height//2, width//2)
        """
        # 主分支计算
        main = self.reduce_conv(x)
        main = self.bn_reduce(main)
        main = self.P_relu(main)

        main = self.main_conv(main)
        main = self.bn_main(main)
        main = self.P_relu(main)

        main = self.expand_conv(main)
        main = self.bn_expand(main)

        # 跳跃分支计算
        skip = self.skip_pool(x)
        skip = self.skip_conv(skip)
        skip = self.bn_skip(skip)

        # 残差连接
        x = main + skip
        x = self.P_relu(x)

        # 训练阶段应用Dropout
        if self.training and self.dropout.p > 0:
            x = self.dropout(x)

        return x


class EnetRegularBottleneck(nn.Module):
    """Enet普通瓶颈模块

    通过1x1卷积降维、3x3普通卷积、1x1卷积升维 序列操作，
    结合残差连接在保持特征图尺寸不变的情况下提取局部特征。
    """

    def __init__(self, in_channels: int, out_channels: int = None, reduction_ratio: int = 4, kernel_size: int = 3, dropout_prob: float = 0.1) -> None:
        """初始化Enet普通瓶颈模块

        参数:
            in_channels:     输入特征图的通道数
            out_channels:    输出特征图的通道数，默认与输入通道数相同
            reduction_ratio: 降维倍率（必须为正整数），中间通道数 = in_channels // reduction_ratio
            kernel_size:     主卷积层的卷积核大小（必须为奇数）
            dropout_prob:    Dropout概率，默认0.1

        异常:
            ValueError: 当降维倍率非正整数或卷积核大小非奇数时抛出
            ValueError: 当降维倍率导致中间通道数为0时抛出
        """
        super().__init__()

        # 设置默认输出通道数
        out_channels = out_channels or in_channels

        # 验证参数有效性
        if not (isinstance(reduction_ratio, int) and reduction_ratio > 0):
            raise ValueError(f"降维倍率必须是正整数，当前值: {reduction_ratio}")
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError(f"卷积核大小必须是≥3的奇数，当前值: {kernel_size}")

        # 计算并验证中间通道数
        reduced_channels = in_channels // reduction_ratio
        if reduced_channels == 0:
            raise ValueError(
                f"降维倍率({reduction_ratio})过大，导致中间通道数为0，输入通道数: {in_channels}"
            )

        # 主分支：降维 -> 普通卷积 -> 升维
        self.reduce_conv = nn.Conv2d(
            in_channels =in_channels     ,
            out_channels=reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(reduced_channels)

        self.main_conv = nn.Conv2d(
            in_channels =reduced_channels,
            out_channels=reduced_channels,
            kernel_size=kernel_size,
            stride=1,  # 保持尺寸不变
            padding=kernel_size // 2,  # 对称填充
            bias=False
        )
        self.bn_main = nn.BatchNorm2d(reduced_channels)

        self.expand_conv = nn.Conv2d(
            in_channels =reduced_channels,
            out_channels=out_channels    ,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_expand = nn.BatchNorm2d(out_channels)

        # 跳跃分支：通道数不匹配时需调整
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels =in_channels ,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.bn_skip = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

        # 激活函数和Dropout
        self.P_relu = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels , height, width)

        返回:
            x: 输出张量，形状为 (batch_size, out_channels, height, width)
        """
        # 主分支计算
        main = self.reduce_conv(x)
        main = self.bn_reduce(main)
        main = self.P_relu(main)

        main = self.main_conv(main)
        main = self.bn_main(main)
        main = self.P_relu(main)

        main = self.expand_conv(main)
        main = self.bn_expand(main)

        # 跳跃分支计算
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
            skip = self.bn_skip(skip)
            x = main + skip
        else:
            x = main + x  # 直接残差连接

        x = self.P_relu(x)

        # 训练阶段应用Dropout
        if self.training and self.dropout.p > 0:
            x = self.dropout(x)

        return x


class EnetDilatedBottleneck(nn.Module):
    """Enet空洞卷积瓶颈模块

    通过1x1卷积降维、空洞卷积（扩大感受野）、1x1卷积升维的序列操作，
    结合残差连接在保持特征图尺寸不变的情况下捕捉长距离上下文信息。
    """

    def __init__(self, in_channels: int, out_channels: int = None, reduction_ratio: int = 4, kernel_size: int = 3, dilation: int = 2, dropout_prob: float = 0.1) -> None:
        """初始化Enet空洞卷积瓶颈模块

        参数:
            in_channels:     输入特征图的通道数
            out_channels:    输出特征图的通道数，默认与输入通道数相同
            reduction_ratio: 降维倍率（必须为正整数），中间通道数 = in_channels // reduction_ratio
            kernel_size:     主卷积层的卷积核大小（必须为奇数）
            dilation:        空洞率（必须≥1的整数），控制感受野大小
            dropout_prob:    Dropout概率，默认0.1

        异常:
            ValueError: 当降维倍率非正整数、空洞率非法或卷积核大小非奇数时抛出
            ValueError: 当降维倍率导致中间通道数为0时抛出
        """
        super().__init__()

        # 设置默认输出通道数
        out_channels = out_channels or in_channels

        # 验证参数有效性
        if not (isinstance(reduction_ratio, int) and reduction_ratio > 0):
            raise ValueError(f"降维倍率必须是正整数，当前值: {reduction_ratio}")
        if not (isinstance(dilation, int) and dilation >= 1):
            raise ValueError(f"空洞率必须是≥1的整数，当前值: {dilation}")
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError(f"卷积核大小必须是≥3的奇数，当前值: {kernel_size}")

        # 计算并验证中间通道数
        reduced_channels = in_channels // reduction_ratio
        if reduced_channels == 0:
            raise ValueError(
                f"降维倍率({reduction_ratio})过大，导致中间通道数为0，输入通道数: {in_channels}"
            )

        # 主分支：降维 -> 空洞卷积 -> 升维
        self.reduce_conv = nn.Conv2d(
            in_channels =in_channels     ,
            out_channels=reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(reduced_channels)

        # 计算空洞卷积的填充量（保持尺寸不变）
        padding = dilation * (kernel_size - 1) // 2
        self.main_conv = nn.Conv2d(
            in_channels =reduced_channels,
            out_channels=reduced_channels,
            kernel_size=kernel_size,
            stride=1,  # 保持尺寸不变
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn_main = nn.BatchNorm2d(reduced_channels)

        self.expand_conv = nn.Conv2d(
            in_channels =reduced_channels,
            out_channels=out_channels    ,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_expand = nn.BatchNorm2d(out_channels)

        # 跳跃分支：通道数不匹配时需调整
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels =in_channels ,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.bn_skip = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

        # 激活函数和Dropout
        self.P_relu = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels , height, width)

        返回:
            x: 输出张量，形状为 (batch_size, out_channels, height, width)
        """
        # 主分支计算
        main = self.reduce_conv(x)
        main = self.bn_reduce(main)
        main = self.P_relu(main)

        main = self.main_conv(main)  # 空洞卷积操作
        main = self.bn_main(main)
        main = self.P_relu(main)

        main = self.expand_conv(main)
        main = self.bn_expand(main)

        # 跳跃分支计算
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
            skip = self.bn_skip(skip)
            x = main + skip
        else:
            x = main + x  # 直接残差连接

        x = self.P_relu(x)

        # 训练阶段应用Dropout
        if self.training and self.dropout.p > 0:
            x = self.dropout(x)

        return x


class EnetAsymmetricBottleneck(nn.Module):
    """Enet非对称卷积瓶颈模块

    通过1x1卷积降维、非对称卷积（kx1 + 1xk）、1x1卷积升维的序列操作，
    在保持感受野的同时减少计算量，适合提取复杂空间特征。
    """

    def __init__(self, in_channels: int, out_channels: int = None, reduction_ratio: int = 4, kernel_size: int = 5, dropout_prob: float = 0.1) -> None:
        """初始化Enet非对称卷积瓶颈模块

        参数:
            in_channels:     输入特征图的通道数
            out_channels:    输出特征图的通道数，默认与输入通道数相同
            reduction_ratio: 降维倍率（必须为正整数），中间通道数 = in_channels // reduction_ratio
            kernel_size:     非对称卷积的等效核大小（必须为≥3的奇数）
            dropout_prob:    Dropout概率，默认0.1

        异常:
            ValueError: 当降维倍率非正整数或卷积核大小非奇数时抛出
            ValueError: 当降维倍率导致中间通道数为0时抛出
        """
        super().__init__()

        # 设置默认输出通道数
        out_channels = out_channels or in_channels

        # 验证参数有效性
        if not (isinstance(reduction_ratio, int) and reduction_ratio > 0):
            raise ValueError(f"降维倍率必须是正整数，当前值: {reduction_ratio}")
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError(f"卷积核大小必须是≥3的奇数，当前值: {kernel_size}")

        # 计算并验证中间通道数
        reduced_channels = in_channels // reduction_ratio
        if reduced_channels == 0:
            raise ValueError(
                f"降维倍率({reduction_ratio})过大，导致中间通道数为0，输入通道数: {in_channels}"
            )

        # 主分支：降维 -> 非对称卷积 -> 升维
        self.reduce_conv = nn.Conv2d(
            in_channels =in_channels     ,
            out_channels=reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(reduced_channels)

        # 非对称卷积（kx1 + 1xk）
        padding = (kernel_size - 1) // 2
        self.asymmetric_conv = nn.Sequential(
            nn.Conv2d(
                in_channels =reduced_channels,
                out_channels=reduced_channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                padding=(padding, 0),
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels =reduced_channels,
                out_channels=reduced_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, padding),
                bias=False
            )
        )
        self.bn_asymmetric = nn.BatchNorm2d(reduced_channels)

        self.expand_conv = nn.Conv2d(
            in_channels =reduced_channels,
            out_channels=out_channels    ,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_expand = nn.BatchNorm2d(out_channels)

        # 跳跃分支：通道数不匹配时需调整
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels =in_channels ,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.bn_skip = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

        # 激活函数和Dropout
        self.P_relu = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels , height, width)

        返回:
            x: 输出张量，形状为 (batch_size, out_channels, height, width)
        """
        # 主分支计算
        main = self.reduce_conv(x)
        main = self.bn_reduce(main)
        main = self.P_relu(main)

        main = self.asymmetric_conv(main)  # 非对称卷积操作
        main = self.bn_asymmetric(main)
        main = self.P_relu(main)

        main = self.expand_conv(main)
        main = self.bn_expand(main)

        # 跳跃分支计算
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
            skip = self.bn_skip(skip)
            x = main + skip
        else:
            x = main + x  # 直接残差连接

        x = self.P_relu(x)

        # 训练阶段应用Dropout
        if self.training and self.dropout.p > 0:
            x = self.dropout(x)

        return x


class EnetUpsampleBottleneck(nn.Module):
    """Enet上采样瓶颈模块

    通过1x1卷积降维、转置卷积上采样、1x1卷积升维的序列操作，
    结合残差连接实现特征图的上采样（尺寸翻倍）和通道数调整。
    """

    def __init__(self, in_channels: int, out_channels: int, reduction_ratio: int = 4, kernel_size: int = 3, dropout_prob: float = 0.1) -> None:
        """初始化Enet上采样瓶颈模块

        参数:
            in_channels:     输入特征图的通道数（解码器输入）
            out_channels:    输出特征图的通道数（上采样后目标通道数）
            reduction_ratio: 降维倍率（必须为正整数），中间通道数 = in_channels // reduction_ratio
            kernel_size:     卷积核和转置卷积核大小（必须为奇数）
            dropout_prob:    Dropout概率，默认0.1

        异常:
            ValueError: 当降维倍率非正整数或卷积核大小非奇数时抛出
            ValueError: 当降维倍率导致中间通道数为0时抛出
        """
        super().__init__()

        # 验证参数有效性
        if not (isinstance(reduction_ratio, int) and reduction_ratio > 0):
            raise ValueError(f"降维倍率必须是正整数，当前值: {reduction_ratio}")
        if kernel_size % 2 == 0:
            raise ValueError(f"卷积核大小必须是奇数，当前值: {kernel_size}")

        # 计算并验证中间通道数
        reduced_channels = in_channels // reduction_ratio
        if reduced_channels == 0:
            raise ValueError(
                f"降维倍率({reduction_ratio})过大，导致中间通道数为0，输入通道数: {in_channels}"
            )

        # 主分支：降维 -> 上采样 -> 升维
        self.reduce_conv = nn.Conv2d(
            in_channels= in_channels     ,
            out_channels=reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(reduced_channels)

        # 转置卷积参数（确保尺寸翻倍）
        padding = (kernel_size - 1) // 2
        self.upsample_conv = nn.ConvTranspose2d(
            in_channels =reduced_channels,
            out_channels=reduced_channels,
            kernel_size=kernel_size,
            stride=2,  # 上采样（尺寸翻倍）
            padding=padding,
            output_padding=1,
            bias=False
        )
        self.bn_upsample = nn.BatchNorm2d(reduced_channels)

        self.expand_conv = nn.Conv2d(
            in_channels =reduced_channels,
            out_channels=out_channels    ,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_expand = nn.BatchNorm2d(out_channels)

        # 跳跃分支：上采样 + 通道调整
        self.skip_upsample = nn.ConvTranspose2d(
            in_channels =in_channels ,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False
        )
        self.bn_skip = nn.BatchNorm2d(out_channels)

        # 激活函数和Dropout
        self.P_relu = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels , height  , width  )

        返回:
            x: 输出张量，形状为 (batch_size, out_channels, height*2, width*2)
        """
        # 主分支计算
        main = self.reduce_conv(x)
        main = self.bn_reduce(main)
        main = self.P_relu(main)

        main = self.upsample_conv(main)  # 转置卷积上采样
        main = self.bn_upsample(main)
        main = self.P_relu(main)

        main = self.expand_conv(main)
        main = self.bn_expand(main)

        # 跳跃分支计算
        skip = self.skip_upsample(x)
        skip = self.bn_skip(skip)

        # 残差连接
        x = main + skip
        x = self.P_relu(x)

        # 训练阶段应用Dropout
        if self.training and self.dropout.p > 0:
            x = self.dropout(x)

        return x


class EnetFinalConv(nn.Module):
    """Enet最终卷积层

    通过转置卷积完成最后一次上采样，将解码器输出转换为与输入图像尺寸相同的
    语义分割结果（每个像素对应类别分数）。
    """

    def __init__(self, in_channels: int, num_classes: int, kernel_size: int = 3, output_size: Tuple[int, int] = None) -> None:
        """初始化Enet最终卷积层

        参数:
            in_channels: 输入特征图的通道数（来自解码器输出）
            num_classes: 分割任务的类别数量
            kernel_size: 转置卷积核大小（必须为奇数）
            output_size: 目标输出尺寸 (height, width)，可选，用于精确匹配输入图像尺寸

        异常:
            ValueError: 当卷积核大小非奇数或output_size格式非法时抛出
        """
        super().__init__()

        # 验证参数有效性
        if kernel_size % 2 == 0:
            raise ValueError(f"卷积核大小必须是奇数，当前值: {kernel_size}")
        if output_size is not None:
            if not (isinstance(output_size, tuple) and len(output_size) == 2 and
                    all(isinstance(s, int) and s > 0 for s in output_size)):
                raise ValueError(f"output_size必须是(高, 宽)格式的正整数元组，当前值: {output_size}")

        self.output_size = output_size
        padding = (kernel_size - 1) // 2

        # 最终转置卷积（上采样+类别预测）
        self.final_conv = nn.ConvTranspose2d(
            in_channels =in_channels,
            out_channels=num_classes,
            kernel_size=kernel_size,
            stride=2,  # 最后一次上采样
            padding=padding,
            output_padding=1,
            bias=True  # 输出层保留偏置
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels, height,        width       )

        返回:
            x: 输出张量，形状为 (batch_size, num_classes, target_height, target_width)
            其中target_height和target_width与输入图像尺寸相同
        """
        x = self.final_conv(x)

        # 精确调整输出尺寸（如需要）
        if self.output_size is not None:
            x = F.interpolate(
                x,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )

        return x


class Enet(nn.Module):
    """完整的Enet语义分割模型

    结合编码器-解码器结构，通过下采样提取特征和上采样恢复分辨率，
    实现端到端的语义分割任务。
    """

    def __init__(self, num_classes: int, in_channels: int = 3, output_size: Tuple[int, int] = None) -> None:
        """初始化Enet模型

        参数:
            num_classes:  分割任务的类别数量
            in_channels:  输入图像的通道数，默认3（RGB图像）
            output_size:  目标输出尺寸 (height, width)，可选，用于精确匹配输入图像尺寸
        """
        super().__init__()

        # 编码器部分
        self.initial_block = EnetInitialBlock(in_channels=in_channels, out_channels=16)

        # 阶段1：下采样到1/2尺寸
        self.downsample1 = EnetDownsampleBottleneck(in_channels=16, out_channels=64, reduction_ratio=4, dropout_prob=0.01)
        self.regular1_1 = EnetRegularBottleneck(in_channels=64, reduction_ratio=4, kernel_size=3)
        self.regular1_2 = EnetRegularBottleneck(in_channels=64, reduction_ratio=4, kernel_size=3)
        self.regular1_3 = EnetRegularBottleneck(in_channels=64, reduction_ratio=4, kernel_size=3)
        self.regular1_4 = EnetRegularBottleneck(in_channels=64, reduction_ratio=4, kernel_size=3)

        # 阶段2：下采样到1/4尺寸
        self.downsample2 = EnetDownsampleBottleneck(in_channels=64, out_channels=128, reduction_ratio=4, dropout_prob=0.1)
        self.regular2_1 = EnetRegularBottleneck(in_channels=128, reduction_ratio=4, kernel_size=3)
        self.dilated2_1 = EnetDilatedBottleneck(in_channels=128, reduction_ratio=4, kernel_size=3, dilation=2)
        self.asymmetric2_1 = EnetAsymmetricBottleneck(in_channels=128, reduction_ratio=4, kernel_size=5)
        self.dilated2_2 = EnetDilatedBottleneck(in_channels=128, reduction_ratio=4, kernel_size=3, dilation=4)
        self.regular2_2 = EnetRegularBottleneck(in_channels=128, reduction_ratio=4, kernel_size=3)
        self.dilated2_3 = EnetDilatedBottleneck(in_channels=128, reduction_ratio=4, kernel_size=3, dilation=8)
        self.asymmetric2_2 = EnetAsymmetricBottleneck(in_channels=128, reduction_ratio=4, kernel_size=5)
        self.dilated2_4 = EnetDilatedBottleneck(in_channels=128, reduction_ratio=4, kernel_size=3, dilation=16)

        # 解码器部分
        # 阶段3：上采样到1/2尺寸
        self.upsample1 = EnetUpsampleBottleneck(in_channels=128, out_channels=64, reduction_ratio=4, kernel_size=3)
        self.regular3_1 = EnetRegularBottleneck(in_channels=64, reduction_ratio=4, kernel_size=3)
        self.regular3_2 = EnetRegularBottleneck(in_channels=64, reduction_ratio=4, kernel_size=3)

        # 阶段4：上采样到原始尺寸
        self.upsample2 = EnetUpsampleBottleneck(in_channels=64, out_channels=16, reduction_ratio=4, kernel_size=3)
        self.regular4_1 = EnetRegularBottleneck(in_channels=16, reduction_ratio=4, kernel_size=3)

        # 最终卷积层（输出类别预测）
        self.final_conv = EnetFinalConv(in_channels=16, num_classes=num_classes, output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入图像张量，形状为 (batch_size, in_channels, height, width)

        返回:
            x: 分割结果张量，形状为 (batch_size, num_classes, height, width)
        """
        # 保存输入尺寸用于最终调整（如果需要）
        input_size = (x.size(2), x.size(3))

        # 编码器前向传播
        x = self.initial_block(x)  # (B, 16, H/2, W/2)

        x = self.downsample1(x)    # (B, 64, H/4, W/4)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        x = self.downsample2(x)    # (B, 128, H/8, W/8)
        x = self.regular2_1(x)
        x = self.dilated2_1(x)
        x = self.asymmetric2_1(x)
        x = self.dilated2_2(x)
        x = self.regular2_2(x)
        x = self.dilated2_3(x)
        x = self.asymmetric2_2(x)
        x = self.dilated2_4(x)

        # 解码器前向传播
        x = self.upsample1(x)      # (B, 64, H/4, W/4)
        x = self.regular3_1(x)
        x = self.regular3_2(x)

        x = self.upsample2(x)      # (B, 16, H/2, W/2)
        x = self.regular4_1(x)

        # 最终输出（恢复到原始尺寸）
        if self.final_conv.output_size is None:
            self.final_conv.output_size = input_size
        x = self.final_conv(x)     # (B, num_classes, H, W)

        return x
