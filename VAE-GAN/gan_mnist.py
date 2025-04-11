import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from information_printer import print_grad

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
latent_dim = 512  # 隐空间维度
lr = 0.0002  # 学习率
batch_size = 64  # 批量大小
epochs = 50  # 训练轮数

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 将像素值归一化到[-1, 1]
])

# MNIST 数据集加载
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(),
            nn.Unflatten(1,(256,7,7)),

            # nn.ConvTranspose2d(256,128,4,2,1),
            nn.Upsample((14,14), mode='bilinear', align_corners=False),
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # nn.Conv2d(32,64,3,1,1),
            # nn.LeakyReLU(),
            # nn.Conv2d(64, 64, 3, 1, 1),
            # nn.LeakyReLU(),

            # nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Upsample((28, 28), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 1, 3, 1, 1),
            # nn.LeakyReLU(),
            # nn.Conv2d(64, 64, 3, 1, 1),
            # nn.LeakyReLU(),
            # nn.Conv2d(64, 1, 1, 1, 0),
            # nn.LeakyReLU(),
            #
            # nn.Flatten(),
            #
            # nn.Linear(784, 1024),
            # nn.LeakyReLU(),
            # nn.Linear(1024, 1024),
            # nn.LeakyReLU(),
            # nn.Linear(1024, 28*28),
            nn.Tanh()  # 使用 Tanh 激活函数，将输出限制在[-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.resize(img.size(0), 1, 28, 28)  # 将图像 reshape 成 MNIST 的格式
        return img

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(7*7*128, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数，将输出限制在[0, 1]
        )

    def forward(self, img):
        # img_flat = img.view(img.size(0), -1)  # 将图像展平为向量
        validity = self.model(img)
        return validity

# 初始化生成器和判别器
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
# discriminator.load_state_dict(torch.load("dis.pth"))

# 损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer_G = optim.Adam(generator.parameters(), lr=1*lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.05*lr)

# 训练GAN
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # 将图像加载到设备
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # 创建标签
        valid = torch.ones(batch_size, 1, device=device)  # 真实图像的标签
        fake = torch.zeros(batch_size, 1, device=device)  # 生成图像的标签

        # 训练生成器



        for k in range(1):
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)  # 随机噪声输入
            gen_imgs = generator(z)  # 生成图像
            g_loss = criterion(discriminator(gen_imgs), valid)  # 计算生成器损失
            print(g_loss)
            g_loss.backward()
            optimizer_G.step()

        # 训练判别器
        for j in range(1):
            if g_loss<18:#生成器损失过大时判别器停止训练
                z = torch.randn(batch_size, latent_dim, device=device)  # 随机噪声输入
                gen_imgs = generator(z)  # 生成图像
                optimizer_D.zero_grad()
                real_loss = criterion(discriminator(real_imgs), valid)  # 真实图像的损失
                fake_loss = criterion(discriminator(gen_imgs.detach()), fake)  # 生成图像的损失
                d_loss = (real_loss + fake_loss) / 2  # 判别器的总损失
                d_loss.backward()
                optimizer_D.step()

    # 打印损失
    print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {d_loss.item():.10f}, Loss G: {g_loss.item():.10f}")

    # 每隔一定轮次展示生成的图像
    if epoch % 1 == 0:
        print_grad(generator)
        torch.save(discriminator.state_dict(), "../Diffusion_Model/dis.pth")
        with torch.no_grad():
            sample_z = torch.randn(64, latent_dim, device=device)
            gen_imgs = generator(sample_z)
            gen_imgs = gen_imgs.resize(gen_imgs.size(0), 1, 28, 28).cpu().numpy()
            fig, axs = plt.subplots(8, 8, figsize=(8, 8))
            for i in range(8):
                for j in range(8):
                    axs[i, j].imshow(gen_imgs[i * 8 + j, 0])
                    axs[i, j].axis('off')
            plt.show()
