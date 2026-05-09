########################################################################################################
import torch
import torch.nn as nn
from ENET_Handle import create_Image_Mask_enet, SimulatedHandle
from Transformer_Encoder import EncoderWithOutHeads
import pandas as pd


class AutoDriveModel(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(AutoDriveModel, self).__init__()
        self.device = device
        self.simulated_handle = SimulatedHandle().to(device)
        self.feature_dim = 128
        self.num_heads = 8
        self.dim_per_head = self.feature_dim // self.num_heads

        self.transformer = EncoderWithOutHeads(
            Enc_num_Linear0=3,
            dim_models=self.feature_dim,
            dim_heads=self.num_heads,
            Feed_dim=2048,
            dim_Q=self.dim_per_head,
            dim_K=self.dim_per_head,
            dim_V=self.dim_per_head,
            Batch_size=1,
            Len=25  # 需与SimulatedHandle输出序列长度匹配
        ).to(device)

    def forward(self, masks_tensor):

        masks_tensor = masks_tensor.to(self.device).float()

        # SimulatedHandle处理
        handle_OUTPUT = self.simulated_handle(masks_tensor)

        # 通过Transformer处理
        OUTPUT0, OUTPUT1, OUTPUT2, OUTPUT3 = self.transformer(handle_OUTPUT)

        # 拼接输出
        result_tensor = torch.cat([OUTPUT0, OUTPUT1, OUTPUT2, OUTPUT3], dim=1)
        return result_tensor

    def Process_image(self, image_mask_path):

        masks_tensor0 = create_Image_Mask_enet(image_mask_path)
        return masks_tensor0


if __name__ == "__main__":
    # 初始化模型
    excel_Path = "D:/AutoDrive/AUTO/Transformer_Encode/CNNS/Auto.xlsx" # PNG, Handle1, Handle2...
    df = pd.read_excel(excel_Path, sheet_name=0, header=None)
    image_path = df.iloc[0, 0]
    controls = df.iloc[0, 1:5].astype(float).values
    model = AutoDriveModel()
    control_tensor = torch.tensor(controls, dtype=torch.float32).unsqueeze(0).to(model.device)
    criterion = nn.MSELoss()  # 根据任务类型调整
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    for epoch in range(1000):
        model.train()
        total_loss = 0
        masks_tensor = model.Process_image(image_path)
        out = model(masks_tensor)
        loss = criterion(out, control_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler_optimizer.step()
        total_loss += loss.item()
        print(total_loss)

    model_save_path = "auto_drive_model.pth"
    torch.save(model.state_dict(), model_save_path)
    # 加载保存模型
    model = AutoDriveModel()
    model.load_state_dict(torch.load("auto_drive_model.pth"))
    model.eval()

    # 验证图片
    image_path = df.iloc[0, 0]
    masks_tensor = model.Process_image(image_path)
    with torch.no_grad():
        out_tensor = model(masks_tensor)

    # 打印张量
    print("模型输出张量:")
    print(out_tensor)
########################################################################################################