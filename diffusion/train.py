# 训练主函数

import config
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
# from models import net
from models_copy import net
import information_printer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs/model_structure')
# 假设你有一个模型和输入
dummy_input = [torch.randn(1, 3, 64, 64).to(config.device),torch.randint(0, 100, (1,)).long().to(config.device)]  # 例如，对于一个输入形状为 [1, 3, 224, 224] 的模型
writer.add_graph(net, dummy_input)


def get_index_from_list(vals, t, x_shape):
    # 直接copy的懒得编了啊哈哈哈哈哈哈
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def noise_schedule(x0,t,device='cpu'):
    # 给定t返回加噪后的图像以及实际噪声（批处理，t是一个batch长度的序列）
    noise = torch.randn_like(x0)#纯高斯噪声,均值为原图像素
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    # 原图和噪声按一定比例混合就是t时刻加噪后的结果
    return sqrt_alphas_cumprod_t.to(device) * x0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# 计算所有所需的中间参数,beta控制加的噪声强度随时间的变化
T = config.T
epochs=config.epochs
device=config.device
betas = torch.linspace(start=0.0001, end=config.linear_beta_end, steps=T)# 获取一个递增的beta列表
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)





def train(previous_path=None, step=0):
    net.train()
    #定义优化器与损失函数
    optimizer=torch.optim.RMSprop(net.parameters(),lr=config.learning_rate)
    criterion=torch.nn.MSELoss()

    start_epo=0
    if previous_path!=None:
        net.load_state_dict(torch.load(previous_path))
        start_epo=1+int(previous_path[-6:-4])

    def get_loss(model, x_0, t):
        x_noisy, noise = noise_schedule(x_0, t, device)
        noise_pred = model(x_noisy, t)
        return criterion(noise_pred,noise)

    # 加载预处理完毕的数据集
    image_data=utils.data

    # 创建数据加载器
    data_loader = DataLoader(image_data, batch_size=config.batch_size, shuffle=config.shuffle)

    running_loss=0.0
    step=0
    # 训练主循环
    for epo in range(start_epo,epochs):
        for batch, _ in data_loader:

            t = torch.randint(0, T, (batch.shape[0],), device=device).long()
            loss = get_loss(net, batch, t)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            step+=1
            running_loss+=loss.item()
            if step%config.step_print_loss==0:
                writer.add_scalar('Training Loss', loss.item(), step)
                print(f"Epoch {epo} step{step} | Loss: {running_loss/config.step_print_loss} ")
                running_loss=0

            if step % config.step_save == 0:
                save_path = "./results/" + "Diffusion_image_size={}".format(config.image_size) + "_step={}.pth".format(step)
                torch.save(net.state_dict(),save_path)
    writer.close()





if __name__ == '__main__':
    # pass
    # train("./results/Diffusion_image_size=64_step=16000.pth")
    train()







