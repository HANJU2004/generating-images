import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from information_printer import print_grad,print_num_parameters

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
latent_dim = 1024  # 隐空间维度
lr = 0.0002  # 学习率
batch_size = 256  # 批量大小
epochs = 200  # 训练轮数

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 将像素值归一化到[-1, 1]
])

# MNIST 数据集加载
train_dataset = datasets.ImageFolder(root="C:/Datasets/animeface64",transform=transform)
# subset_indices = torch.arange(0, 128)  # 选择前x个样本
# subset = Subset(train_dataset, subset_indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)


# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(latent_dim, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            # nn.Linear(512, 1024*4*4),
            # nn.BatchNorm1d(1024*4*4),
            # nn.LeakyReLU(),
            # nn.Unflatten(1,(1024,4,4)),

            nn.Unflatten(1, (latent_dim, 1, 1)),

            nn.ConvTranspose2d(latent_dim,1024,4,1,0,bias=False),
            nn.BatchNorm2d(1024),
            nn.Tanh(),

            nn.ConvTranspose2d(1024,512,4,2,1),
            # nn.Upsample((8,8), mode='bilinear', align_corners=False),
            # nn.Conv2d(1024,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.Tanh(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            # nn.Upsample((16, 16), mode='bilinear', align_corners=False),
            # nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Tanh(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # nn.Upsample((32, 32), mode='bilinear', align_corners=False),
            # nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            # nn.Upsample((64, 64), mode='bilinear', align_corners=False),
            # nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()  # 使用 Tanh 激活函数，将输出限制在[-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.resize(img.size(0), 3, 64, 64)
        return img

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(2,2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(8*8*256, 1),
            #TODO:youhuazhege
            nn.Sigmoid()  # 使用 Sigmoid 激活函数，将输出限制在[0, 1]
        )

    def forward(self, img):
        # img_flat = img.view(img.size(0), -1)  # 将图像展平为向量
        validity = self.model(img)
        return validity

# 初始化生成器和判别器
generator = Generator(latent_dim).to(device)
print_num_parameters(generator)
# generator.load_state_dict(torch.load("weights/gen_epo_66.pth"))
discriminator = Discriminator().to(device)
print_num_parameters(discriminator)
# discriminator.load_state_dict(torch.load("weights/dis_epo_66.pth"))
current_epoch=0
# 损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer_G = optim.Adam(generator.parameters(), lr=1*lr)
optimizer_D = optim.SGD(discriminator.parameters(), lr=1*lr,momentum=0.8)

sample_z = torch.randn(64, latent_dim, device=device)

if __name__ == '__main__':# 训练GAN


    for epoch in range(current_epoch,epochs):
        for i, (imgs, _) in enumerate(train_loader):
            # plt.imshow(imgs[0].cpu().numpy().transpose(1,2,0))
            # plt.show()
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
                print("G: ",g_loss.item())
                g_loss.backward()
                optimizer_G.step()

            # 训练判别器
            for j in range(1):
                z = torch.randn(batch_size, latent_dim, device=device)  # 随机噪声输入
                gen_imgs = generator(z)  # 生成图像
                optimizer_D.zero_grad()
                real_loss = criterion(discriminator(real_imgs), valid)  # 真实图像的损失
                fake_loss = criterion(discriminator(gen_imgs.detach()), fake)  # 生成图像的损失
                d_loss = (real_loss + fake_loss) / 2  # 判别器的总损失
                print("D: ",d_loss.item())
                d_loss.backward()
                optimizer_D.step()

        # 打印损失
        print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {d_loss.item():.10f}, Loss G: {g_loss.item():.10f}")

        if epoch%6==0:
            torch.save(generator.state_dict(),f"weights/gen_epo_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"weights/dis_epo_{epoch}.pth")

        # 每隔一定轮次展示生成的图像
        if epoch % 12 == 0:
            print_grad(generator)
            with torch.no_grad():

                gen_imgs = generator(sample_z)
                gen_imgs = gen_imgs.cpu().numpy().transpose(0,2,3,1)
                gen_imgs=(gen_imgs+1)/2
                fig, axs = plt.subplots(8, 8, figsize=(8, 8))
                for i in range(8):
                    for j in range(8):
                        axs[i, j].imshow(gen_imgs[i * 8 + j])

                plt.show()