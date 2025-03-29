import numpy as np
import matplotlib.pyplot as plt
import torch

import src.algorithms.richardson_lucy as rl
import src.noise.noise as noise
import src.blur.gaussian_blur as gaussian_blur
from PIL import Image

device = "cuda"

k_size = 61
k_std = 6.0
n_std = 0.1
n_rate = 0.1
k_intensity = 0.2

# Load the data
ref = Image.open('img/img_88.png').convert('L')

ref = torch.from_numpy(np.array(ref)).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
ref = ref[:, :, :-1, :-1]

gaussianblur = gaussian_blur.GaussianBlur(k_size, k_std).to(device)
k_ref = gaussianblur.get_kernel().view(1, 1, k_size, k_size).to(device)

# motionblur = motion_blur.MotionBlur(k_size, k_intensity).to(device)
# k_ref = motionblur.get_kernel().view(1, 1, k_size, k_size).to(device)

noiser = noise.PoissonNoise(n_rate)
# noiser = noise.GaussianNoise(n_std)

# Blur the image
y = gaussianblur(ref)
# y = motionblur(ref)

# Add noise
y = noiser(y)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(y[0, 0].detach().cpu().numpy(), cmap='gray')
plt.subplot(1, 2, 2)   
plt.imshow(k_ref[0, 0].detach().cpu().numpy(), cmap='gray')
plt.show()

x_0 = torch.ones_like(y) 

res = rl.richardson_lucy(y, x_0, k_ref, steps=100, tv=False)

plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.imshow(y[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Noisy and blurred image', fontsize=16)
plt.subplot(1, 3, 2)
plt.imshow(res[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Restored image', fontsize=16)
plt.subplot(1, 3, 3)
plt.imshow(ref[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Original image', fontsize=16)
plt.show()

x_0 = torch.ones_like(y) 

res_tv = rl.richardson_lucy(y, x_0, k_ref, steps=100, tv=True)

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(y[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Noisy and blurred image', fontsize=16)
plt.subplot(1, 4, 2)
plt.imshow(res[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Restored image without TV', fontsize=16)
plt.subplot(1, 4, 3)
plt.imshow(res_tv[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Restored image with TV', fontsize=16)
plt.subplot(1, 4, 4)
plt.imshow(ref[0, 0].detach().cpu().numpy(), cmap='gray')
plt.title('Original image', fontsize=16)
plt.show()