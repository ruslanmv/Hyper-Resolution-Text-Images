# How to Perform Super Resolution on Low-Resolution Text Images using GANs (SRGAN/ESRGAN)

In this blog, we will walk through the process of performing super-resolution on an image containing unstructured text with low resolution using Generative Adversarial Networks (GANs). Specifically, we will explore the Super-Resolution GAN (SRGAN) and its enhanced version, ESRGAN. These models can upscale images by generating finer textures and achieving more photorealistic results.

## What is Super-Resolution GAN?

SRGAN is a deep learning-based technique designed to enhance the resolution of images. It consists of two networks:

1. **Generator**: This network takes a low-resolution image as input and outputs a super-resolved image.
2. **Discriminator**: This network ensures that the generated high-resolution image looks as realistic as possible by classifying it as real or fake.

The ESRGAN is an enhanced version of SRGAN. It improves the performance by refining the architecture of both the generator and the discriminator.

In this tutorial, we will implement an image super-resolution pipeline using SRGAN or ESRGAN and train the model to upscale a low-resolution image.

---

## Steps to Perform Super-Resolution using SRGAN/ESRGAN

### Step 1: Install Required Libraries

Before starting, we need to install the necessary Python libraries. These include PyTorch (or TensorFlow if you prefer), NumPy, Matplotlib, and other image processing libraries.

```bash
pip install torch torchvision numpy matplotlib opencv-python
```

### Step 2: Load and Preprocess the Image

Next, we will load the image containing unstructured text, convert it to grayscale (since text is often monochrome), and resize it to simulate low resolution.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the low-resolution image
image_path = 'low_res_text_image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.imshow(image, cmap='gray')
plt.title('Low-Resolution Image')
plt.show()

# Resize image to simulate low resolution
low_res_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
plt.imshow(low_res_image, cmap='gray')
plt.title('Simulated Low-Resolution Image')
plt.show()
```

### Step 3: Define the SRGAN Model

We will now implement the architecture for SRGAN or ESRGAN. Below is an example of a simplified SRGAN architecture in PyTorch. You can replace it with a more advanced ESRGAN architecture for better performance.

#### Generator Network

The generator uses a deep convolutional neural network to upscale the image.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class Generator(nn.Module):
    def __init__(self, channels=1, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        # Add residual blocks
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(64))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsample layers
        self.upsample1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        residual = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.add(out, residual)
        out = self.upsample1(out)
        out = nn.functional.pixel_shuffle(out, 2)
        out = self.upsample2(out)
        return out
```

#### Discriminator Network

The discriminator checks whether the generated high-resolution image is real or fake.

```python
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        in_channels, height, width = input_shape

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(512 * (height // 4) * (width // 4), 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        return torch.sigmoid(self.fc(out))
```

### Step 4: Training the SRGAN/ESRGAN

Now that we have both the generator and discriminator, we can start training the model. We will use a dataset of low- and high-resolution images and iteratively train the GAN.

```python
# Define loss function and optimizers
adversarial_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

generator = Generator().cuda()
discriminator = Discriminator((1, 64, 64)).cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        
        # Load real images (high-resolution)
        real_imgs = imgs.cuda()
        
        # Generate fake images (super-resolved from low-resolution)
        gen_imgs = generator(low_res_imgs)

        # Adversarial ground truths
        valid = torch.ones((imgs.size(0), 1)).cuda()
        fake = torch.zeros((imgs.size(0), 1)).cuda()

        # Train the Generator
        optimizer_G.zero_grad()
        
        loss_G = adversarial_loss(discriminator(gen_imgs), valid) + mse_loss(gen_imgs, real_imgs)
        loss_G.backward()
        optimizer_G.step()

        # Train the Discriminator
        optimizer_D.zero_grad()

        loss_real = adversarial_loss(discriminator(real_imgs), valid)
        loss_fake = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch}/{num_epochs}] - Loss_G: {loss_G.item()}, Loss_D: {loss_D.item()}")
```

### Step 5: Evaluating and Visualizing the Results

Finally, we evaluate the performance of our generator by passing the low-resolution image through the trained network and visualizing the results.

```python
# Load the pre-trained generator
generator.eval()

# Pass the low-resolution image through the generator
with torch.no_grad():
    low_res_image_tensor = torch.FloatTensor(low_res_image).unsqueeze(0).unsqueeze(0).cuda()
    super_res_image = generator(low_res_image_tensor).squeeze().cpu().numpy()

# Display the super-resolved image
plt.imshow(super_res_image, cmap='gray')
plt.title('Super-Resolution Image')
plt.show()
```

---

## Conclusion

In this blog, we covered how to implement super-resolution on an image containing unstructured low-resolution text

 using GANs. We implemented both the generator and discriminator of SRGAN (or ESRGAN) and used PyTorch to train and evaluate the models. By applying these techniques, we can significantly improve the quality of low-resolution text images.

Feel free to experiment with more advanced architectures, such as ESRGAN, for even better results!