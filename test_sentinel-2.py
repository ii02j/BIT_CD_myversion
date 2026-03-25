import os
import torch
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from models.networks import define_G
from utils import get_device
import argparse

# 1. 定义数据集类（读哨兵切片）
class SentinelDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, img_size=256, bands=[3,2,1]):  # bands默认RGB顺序
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.img_size = img_size
        self.bands = bands  
        self.files = [f for f in os.listdir(t1_dir) if f.endswith('.tif')]
        # 确保两期文件名一一对应
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        t1_path = os.path.join(self.t1_dir, filename)
        t2_path = os.path.join(self.t2_dir, filename)

        # 读影像，取指定波段
        with rasterio.open(t1_path) as src:
            img1 = src.read(self.bands)  # shape (bands, H, W)
        with rasterio.open(t2_path) as src:
            img2 = src.read(self.bands)

        # 转成 [C,H,W] -> 归一化到 [-1,1]
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())   
        img1 = (img1 - 0.5) / 0.5  
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())   
        img2 = (img2 - 0.5) / 0.5 
        print(img1.min(),img1.max())
        print(img2.min(),img2.max())

        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        return {'A': img1, 'B': img2, 'name': filename}

# 2. 加载模型
def load_model(checkpoint_path, device, args):
    model = define_G(args)  # 用你训练时的参数
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    print("checkpoint keys:", checkpoint.keys()) 
    model.load_state_dict(checkpoint['model_G_state_dict'])
    model.to(device)
    model.eval()
    return model

# 3. 推理并保存
def predict_and_save(model, dataloader, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch in dataloader:
            name = batch['name'][0]
            img1 = batch['A'].to(device)
            img2 = batch['B'].to(device)
            # 模型输入形状 [B, C, H, W]
            output = model(img1, img2)  # [B, 2, H, W]
            # 取变化类概率
            prob_change = torch.softmax(output, dim=1)[:, 1, :, :]  # [B, H, W]
            # 二值化（阈值0.5）
            pred = (prob_change > 0.5).float().cpu().numpy().squeeze()

            # 保存为 PNG
            plt.imsave(os.path.join(output_dir, f'{name}_change.png'), pred, cmap='gray')
        
        print("output shape:", output.shape)
        print("output min/max:", output.min().item(), output.max().item())
        print("prob_change min/max:", prob_change.min().item(), prob_change.max().item())
        print("pred sum (变化像素数):", pred.sum())
        plt.imsave(os.path.join(output_dir, f'{name}_prob.png'), prob_change[0].cpu().numpy(), cmap='jet')

if __name__ == '__main__':

    # 文件路径要自己修改
    t1_dir = r"C:\Users\Alex.zeng\Desktop\BIT_CD\sentinel-2\A"   
    t2_dir = r"C:\Users\Alex.zeng\Desktop\BIT_CD\sentinel-2\B"
    checkpoint_path = r"C:\Users\Alex.zeng\Desktop\BIT_CD\checkpoints\CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_trainval_test_200_linear\best_ckpt.pt"  
    output_dir = r"C:\Users\Alex.zeng\Desktop\BIT_CD\result"

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 构造参数（与训练时一致）
    class Args:
        net_G = 'base_transformer_pos_s4_dd8'
        gpu_ids = [0] if device.type == 'cuda' else [-1]
    args = Args()

    # 数据集
    dataset = SentinelDataset(t1_dir, t2_dir, bands=[3,2,1])  # 用RGB，Sentinel-2波段顺序 B04,B03,B02
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in dataloader:
        img1 = batch['A']
        print("Sentinel2 sample - min:", img1.min(), "max:", img1.max(), "mean:", img1.mean())
        break

    # 加载模型
    model = load_model(checkpoint_path, device, args)

    # 预测
    predict_and_save(model, dataloader, output_dir, device)