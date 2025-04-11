import numpy as np
import torch
from models_copy import net
import config
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

save_path = 'results/' + "Diffusion_image_size={}".format(config.image_size) + "_step={}.pth".format(1000)
net.load_state_dict(torch.load(save_path))
net.eval()



betas = torch.linspace(0.0001, 0.02, 1000, dtype=torch.float32)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(config.device)

def get_variance(t):
    prev_t = t-1

    alpha_prod_t = alphas_cumprod[t]
    alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
    current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    # and sample from it to get previous sample
    # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

    # we always take the log of variance, so clamp it to ensure it's not 0
    variance = torch.clamp(variance, min=1e-20)
    return variance


def step(
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    t = timestep
    prev_t = t-1

    # 1. compute alphas, betas
    alpha_prod_t = alphas_cumprod[t]
    alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 3. Clip or threshold "predicted x_0"

    pred_original_sample = pred_original_sample.clamp(-1,1)

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # 6. Add noise
    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = torch.randn_like(model_output).to(device)

        variance = (get_variance(t) ** 0.5) * variance_noise

    pred_prev_sample = pred_prev_sample + variance


    return pred_prev_sample


@torch.no_grad()
def generate():
    img_size = config.image_size
    image = torch.randn((4, 3, img_size, img_size), device=config.device)
    for i in range(config.T)[::-1]:
        t = torch.full((1,), i, device=config.device, dtype=torch.long)
        # 1. predict noise model_output
        model_output = net(image, t)

        # 2. compute previous image: x_t -> x_t-1
        image = step(model_output, t, image)


    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

    return image


if __name__ == '__main__':
    drawing=generate()

    for i in range(0,4):
        plt.imshow(drawing[i])
        plt.show()

