import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 超参数
img_size=64
batch_size = 32
latent_dim = 256
learning_rate = 0.0004
epochs = 10
device="cuda"

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 将数据转换为Tensor
    # 归一化到 [0, 1]
])
data_transforms = transforms.Compose([
    transforms.Resize(img_size),        # 调整图像大小
    transforms.CenterCrop(img_size),    # 中心裁剪
    transforms.RandomHorizontalFlip(),      #随机水平翻转
    transforms.ToTensor(),         # 转换为Tensor
    # transforms.Lambda(lambda t: (t * 2) - 1),  # 缩放至 [-1, 1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
    # transforms.Normalize([0.5], [0.5])
])

# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载预处理完毕的数据集
# train_dataset= datasets.ImageFolder("./fltest",transform=data_transforms)
train_dataset=datasets.ImageFolder(root="C:/Datasets/animeface64",transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),


            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512,2*latent_dim,4,1,0),
            nn.Flatten(),
            # nn.Linear(32*8*8, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 2 * latent_dim)  # 输出均值和方差
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),

            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.Tanh(),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
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

            nn.Sigmoid()  # 保证输出在 [0, 1]
        )


    # 将采样操作变得可微分
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x = x.view(-1, 3*64*64)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)  # 均值和对数方差
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 定义损失函数
def loss_function(x_recon, x, mu, logvar):
    # 注意：x需要是 [0, 1] 范围
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# 初始化模型和优化器
model = VAE().to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))
# model.load_state_dict(torch.load("weights/VAE_img_size=64_step=1000.pth"))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(step=0):
    # 训练模型
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            plt.imshow(data[0].cpu().numpy().transpose(1,2,0))
            plt.show()
            data=data.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data)
            plt.imshow(x_recon[0].detach().cpu().numpy().transpose(1, 2, 0))
            plt.show()
            loss = loss_function(x_recon, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            step+=1
            if step % 1000 ==0:
                torch.save(model.state_dict(),f"weights/VAE_img_size=64_step={step}.pth")

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

def eval():
    # 测试生成
    model.eval()
    with torch.no_grad():
        z = torch.randn(10, latent_dim).to(device)
        samples = model.decoder(z).to("cpu").view(-1,3,img_size,img_size).permute(0,2,3,1)
        # 此处可加入代码可视化生成的样本
        for i in range(samples.shape[0]):
            plt.imshow(samples[i])
            plt.show()

if __name__ == '__main__':

    train()
    eval()