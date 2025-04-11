# 训练完毕后生成用

import numpy as np
import torch
from models_copy import net
import config
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from safetensors.torch import load_file

from diffusers import UNet2DModel

# net = UNet2DModel(
#     sample_size=config.image_size,  # the target image resolution
#     in_channels=3,  # the number of input channels, 3 for RGB images
#     out_channels=3,  # the number of output channels
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
#     down_block_types=(
#         "DownBlock2D",  # a regular ResNet downsampling block
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",
#         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#         "DownBlock2D",
#     ),
#     up_block_types=(
#         "UpBlock2D",  # a regular ResNet upsampling block
#         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#     ),
# )

save_path = 'results/' + "Diffusion_image_size={}".format(config.image_size) + "_epoch={}.pth".format(1000)
net.load_state_dict(torch.load(save_path))
# path='C:/Users/admin/PycharmProjects/AI/Diffusion_offical/ddpm-flowers-128/unet/diffusion_pytorch_model.safetensors'
# net.load_state_dict(load_file(path,device='cuda'))
# net.to(device='cuda')
print("Num params: ", sum(p.numel() for p in net.parameters()))

def get_index_from_list(vals, t, x_shape):
    # 直接copy的懒得编了啊哈哈哈哈哈哈
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# 计算所有所需的中间参数,beta控制加的噪声强度随时间的变化
T = config.T
device=config.device
betas = torch.linspace(start=0.0001, end=config.linear_beta_end, steps=T)# 获取一个递增的beta列表
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

@torch.no_grad()
def denoise_one_timestep(x,t):
    # 进行一个时间步的去噪，返回去噪但又加上了一点噪声的图像（我仍然不太理解这点）
    # 一个可能的猜测：t-1时刻和t时刻的加噪都是用x0和上划线alpha做比例混合，使用以上的参数刚好能准确的回到t-1时刻，而不是直接退回到x0
    # 后续我可以尝试将beta固定为定值再进行公式推导。

    # 从列表中取4个参数（奇怪他为何不直接拿中括号呢）
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    # 按照原文公式进行计算
    output=net(x,t)[0]

    temp=sqrt_recip_alphas_t*(x-output*betas_t/sqrt_one_minus_alphas_cumprod_t)
    if t==0:
        return temp
    else:
        # 再掺入一个额外噪声（按理说是比刚刚减去的噪声略小一些）
        additional_noise=torch.randn_like(x)
        return temp+torch.sqrt(posterior_variance_t)*additional_noise

@torch.no_grad()
def generate():
    # 采样一个纯高斯噪声，反向迭代T次
    # Sample noise
    img_size = config.image_size
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(30, 30))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = denoise_one_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()

generate()