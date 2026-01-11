import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# 1) U-Net 的基本模块：ConvBlock
# -----------------------------
class ConvBlock(nn.Module):
    """
    (Conv -> ReLU) * 2
    保持 H, W 不变，只改变通道数
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        #ConvBlock = 在不改变分辨率的前提下
        # 通过两次 3×3 卷积 + ReLU
        # 把原始像素变成更有判别力的特征表示。
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# 2) 教学级 U-Net
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, base_ch=32):
        super().__init__()

        # Encoder: 下采样
        self.enc1 = ConvBlock(in_ch, base_ch)          # 3 -> 32
        self.enc2 = ConvBlock(base_ch, base_ch * 2)    # 32 -> 64
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)# 64 -> 128

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = ConvBlock(base_ch * 4, base_ch * 8) # 128 -> 256

        # Decoder: 上采样
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)  # 拼接后通道翻倍

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # 输出层：1x1 Conv 做逐像素分类
        self.out = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)          #  [B, 32, H, W]
        x2 = self.enc2(self.pool(x1))  # [B, 64, H/2, W/2]
        x3 = self.enc3(self.pool(x2))  # [B,128, H/4, W/4]

        # Middle
        xm = self.mid(self.pool(x3))   # [B,256, H/8, W/8]

        # Decoder + Skip concat
        y3 = self.up3(xm)              # [B,128, H/4, W/4]
        y3 = torch.cat([y3, x3], dim=1)# [B,256, H/4, W/4]
        y3 = self.dec3(y3)             # [B,128, H/4, W/4]

        y2 = self.up2(y3)              # [B, 64, H/2, W/2]
        y2 = torch.cat([y2, x2], dim=1)# [B,128, H/2, W/2]
        y2 = self.dec2(y2)             # [B, 64, H/2, W/2]

        y1 = self.up1(y2)              # [B, 32, H, W]
        y1 = torch.cat([y1, x1], dim=1)# [B, 64, H, W]
        y1 = self.dec1(y1)             # [B, 32, H, W]

        logits = self.out(y1)          # [B, num_classes, H, W]
        return logits


# -----------------------------
# 3) 数据集：Oxford-IIIT Pet
#    它给的是 trimap (1,2,3)，我们把“宠物”当作 1，“背景”当作 0
# -----------------------------
def preprocess_target(trimap: torch.Tensor) -> torch.Tensor:
    """
    trimap: [H, W]，值通常是 1=pet, 2=border, 3=background (不同实现会有差异)
    这里我们把 1/2 都视为 pet, 3 视为 background
    """
    # 保守写法：<=2 当 pet
    return (trimap <= 2).long()

def show_pred(img, mask, pred, save_path=None):
    """
    img: [3,H,W], mask/pred: [H,W]
    """
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = pred.cpu().numpy()

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1); plt.title("Image"); plt.imshow(img_np); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("GT Mask"); plt.imshow(mask_np, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Pred Mask"); plt.imshow(pred_np, cmap="gray"); plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 输入大小统一一下，训练更稳定
    img_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    target_tf = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),  # -> [1,H,W], uint8
    ])

    train_ds = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="segmentation",
        transform=img_tf,
        target_transform=target_tf,
        download=True,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    model = UNet(in_ch=3, num_classes=2, base_ch=32).to(device)

    # 损失：逐像素的交叉熵
    # logits: [B,2,H,W], target: [B,H,W] (0/1)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("runs", exist_ok=True)

    model.train()
    step = 0
    for epoch in range(3):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
        for img, trimap in pbar:
            img = img.to(device)
            trimap = trimap.squeeze(1).to(device)      # [B,H,W]
            mask = preprocess_target(trimap)           # [B,H,W] 0/1

            logits = model(img)
            loss = criterion(logits, mask)

            optim.zero_grad()
            loss.backward()
            optim.step()

            step += 1
            pbar.set_postfix(loss=float(loss.item()))

            # 每隔 200 step 可视化一次
            if step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    sample_img = img[0]
                    sample_mask = mask[0]
                    pred = torch.argmax(logits[0], dim=0)
                    show_pred(sample_img, sample_mask, pred, save_path=f"runs/step_{step}.png")
                model.train()

    print("Training done. Check runs/*.png for predictions.")

if __name__ == "__main__":
    main()
