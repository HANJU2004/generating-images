# 数据导入与处理定义
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import config
# from datasets import load_dataset


# 数据预处理操作
data_transforms = transforms.Compose([
    transforms.Resize(config.image_size),        # 调整图像大小
    transforms.CenterCrop(config.image_size),    # 中心裁剪
    transforms.RandomHorizontalFlip(),      #随机水平翻转
    transforms.ToTensor(),         # 转换为Tensor
    transforms.Lambda(lambda t: (t * 2) - 1),  # 缩放至 [-1, 1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像归一化
    # transforms.Normalize([0.5], [0.5])
])

#

# 测试用展示
def show_images(data, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(np.transpose(img[0],(1,2,0)))
    plt.show()


data = datasets.ImageFolder(config.data_root,transform=data_transforms)
# dataset_name = "huggan/smithsonian_butterflies_subset"
# data = datasets.ImageFolder('C:/Users/admin/PycharmProjects/AI/Diffusion_Model/flowers-102',transform=data_transforms)
show_images(data)

