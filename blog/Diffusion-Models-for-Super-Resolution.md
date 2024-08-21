# Blog: Super-Resolution on Textual Images using Diffusion Models in Python

## Introduction

Super-resolution is a powerful computer vision technique aimed at enhancing the resolution of low-resolution images. This process is particularly useful for images that contain unstructured text or intricate details. In this blog, we will explore two state-of-the-art diffusion-based models for performing super-resolution:

1. **SR3 (Super-Resolution via Repeated Refinement)** - A diffusion-based model that gradually enhances the quality of an image through multiple refinement steps.
2. **Stable Diffusion for Super-Resolution** - Originally developed for text-to-image generation, stable diffusion models can be adapted to super-resolve images.

In this tutorial, we will implement both models in Python, using relevant libraries, step by step.

## Pre-requisites

Before we begin, make sure you have the following libraries installed in your Python environment:

```bash
pip install torch torchvision diffusers transformers
```

We will use PyTorch for deep learning model implementations, and the `diffusers` library, which provides an excellent collection of diffusion-based models, including SR3 and Stable Diffusion.

## Step 1: Loading Dependencies

We begin by importing the necessary dependencies for our implementation:

```python
import torch
from diffusers import StableDiffusionPipeline, UNet2DModel, DDPMScheduler
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
```

- **StableDiffusionPipeline**: This allows us to work with stable diffusion models.
- **UNet2DModel**: UNet architecture for SR3 model implementation.
- **DDPMScheduler**: To schedule the diffusion process for super-resolution.
- **CLIPProcessor and CLIPModel**: For working with Stable Diffusion and generating enhanced images.

## Step 2: Loading and Preprocessing the Image

Let's load a low-resolution image that contains unstructured text for super-resolution.

```python
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transforms.Resize((64, 64))(image)  # resize to low-resolution size
    return image

# Load the low-resolution text image
low_res_image = load_image('low_res_text_image.png')

# Display the low-resolution image
plt.imshow(low_res_image)
plt.title("Low-Resolution Image")
plt.axis('off')
plt.show()
```

In this step:
- We load the image from a file path and resize it to 64x64 pixels to simulate low resolution.
- The image is displayed using `matplotlib` for visualization.

## Step 3: Implementing SR3 (Super-Resolution via Repeated Refinement)

SR3 is a diffusion model designed specifically for image super-resolution. It uses repeated refinement through forward and reverse diffusion processes.

Here, we implement the SR3 super-resolution process:

```python
class SR3SuperResolution:
    def __init__(self):
        self.model = UNet2DModel.from_pretrained("google/sr3_64_128")
        self.scheduler = DDPMScheduler.from_pretrained("google/sr3_64_128")
    
    def super_resolve(self, low_res_image, steps=1000):
        # Process the low-resolution image for model input
        img = transforms.ToTensor()(low_res_image).unsqueeze(0)
        
        # Generate super-resolution through multiple diffusion steps
        for step in range(steps):
            img = self.scheduler.step(self.model(img, step))
        return transforms.ToPILImage()(img.squeeze(0))

# Perform SR3 super-resolution on the low-resolution image
sr3_model = SR3SuperResolution()
high_res_image_sr3 = sr3_model.super_resolve(low_res_image)

# Display the super-resolved image
plt.imshow(high_res_image_sr3)
plt.title("Super-Resolved Image (SR3)")
plt.axis('off')
plt.show()
```

### Key Steps:
1. **Model Loading**: We load the pre-trained SR3 model and the scheduler that manages the diffusion process.
2. **Image Processing**: The low-resolution image is converted to a tensor and passed through the model for refinement over a number of steps.
3. **Super-Resolution Generation**: After the iterations, the image is converted back to the PIL format and displayed.

## Step 4: Stable Diffusion for Super-Resolution

Stable Diffusion models are generally used for text-to-image generation but can be adapted for super-resolution tasks by modifying the image conditioning.

```python
class StableDiffusionSuperResolution:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        self.pipe.to("cuda")  # Use GPU if available

    def super_resolve(self, low_res_image):
        # Transform low-resolution image to a latent representation
        latent_representation = self.pipe.vae.encode(transforms.ToTensor()(low_res_image).unsqueeze(0).to("cuda"))
        
        # Pass through the model to get a higher-resolution version
        with torch.no_grad():
            high_res_image = self.pipe.decode(latent_representation.sample())
        
        # Convert the result back to image format
        high_res_image = high_res_image[0].cpu().permute(1, 2, 0).numpy()
        high_res_image = (high_res_image * 255).astype("uint8")
        return Image.fromarray(high_res_image)

# Perform Stable Diffusion-based super-resolution
stable_diff_model = StableDiffusionSuperResolution()
high_res_image_stable_diff = stable_diff_model.super_resolve(low_res_image)

# Display the super-resolved image
plt.imshow(high_res_image_stable_diff)
plt.title("Super-Resolved Image (Stable Diffusion)")
plt.axis('off')
plt.show()
```

### Key Steps:
1. **Pipeline Loading**: We use the `StableDiffusionPipeline` to load the pre-trained Stable Diffusion model.
2. **Latent Representation**: The low-resolution image is encoded into a latent representation, which is then decoded to produce a higher-resolution image.
3. **Super-Resolution Generation**: The output is processed to transform back into a usable image format and is displayed.

## Step 5: Comparing Results

Now that we have super-resolved images from both SR3 and Stable Diffusion models, let’s display them side by side for comparison:

```python
# Display low-res, SR3, and Stable Diffusion results side by side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Low-resolution image
axs[0].imshow(low_res_image)
axs[0].set_title("Low-Resolution Image")
axs[0].axis('off')

# SR3 result
axs[1].imshow(high_res_image_sr3)
axs[1].set_title("Super-Resolved (SR3)")
axs[1].axis('off')

# Stable Diffusion result
axs[2].imshow(high_res_image_stable_diff)
axs[2].set_title("Super-Resolved (Stable Diffusion)")
axs[2].axis('off')

plt.show()
```

This step will display the original low-resolution image, the super-resolved image using SR3, and the one generated by Stable Diffusion.

## Conclusion

In this blog, we explored how to use two state-of-the-art diffusion models, **SR3** and **Stable Diffusion**, for super-resolution tasks, particularly focusing on images containing unstructured text. We implemented both models using Python and compared the results.

Diffusion models have proven to be effective in gradually improving image quality through refinement, and they can be applied in various real-world tasks such as text-based image enhancement, document restoration, and more.

Feel free to experiment with different images and model settings to see the full potential of these approaches!