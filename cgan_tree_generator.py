"""
A Conditional Generative Adversarial Network (CGAN) for generating tree images based on age 
and noise inputs.

Key Components:
    Generator Class:
        Takes noise and age as inputs and generates images.
        Uses a series of linear layers and Leaky ReLU activations to transform the input into an image.
    Discriminator Class:
        Evaluates the generated images to determine if they are real or fake.
        Also takes age as an input along with the image.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, age_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + age_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, age):
        x = torch.cat((noise, age), dim=1)
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim, age_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim + age_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, age):
        x = torch.cat((img, age), dim=1)
        return self.model(x)




# Hyperparameters
img_size = 128  # Image dimensions (128x128)
img_dim = img_size * img_size * 3  # Flattened image dimensions
noise_dim = 100  # Noise vector size
age_dim = 1  # One-hot encoded age
batch_size = 32
epochs = 5000
learning_rate = 0.0002

# Initialize models
generator = Generator(noise_dim, age_dim, img_dim)
discriminator = Discriminator(img_dim, age_dim)

# Optimizers and Loss
opt_gen = optim.Adam(generator.parameters(), lr=learning_rate)
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# Data loader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Data augmentation with transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values
])

# Load dataset with augmentation
dataset = ImageFolder(root=r"Code\Tree Age Predictor & Generator\tree_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training
device = "cuda" if torch.cuda.is_available() else "cpu"
generator.to(device)
discriminator.to(device)

for epoch in range(epochs):
    for real_imgs, labels in dataloader:
        real_imgs, labels = real_imgs.view(real_imgs.size(0), -1).to(device), labels.to(device)
        batch_size = real_imgs.size(0)
        
        # Train Discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_imgs = generator(noise, labels.unsqueeze(1).float())
        real_preds = discriminator(real_imgs, labels.unsqueeze(1).float())
        fake_preds = discriminator(fake_imgs.detach(), labels.unsqueeze(1).float())
        
        real_loss = criterion(real_preds, torch.ones_like(real_preds))
        fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
        disc_loss = (real_loss + fake_loss) / 2
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Train Generator
        gen_preds = discriminator(fake_imgs, labels.unsqueeze(1).float())
        gen_loss = criterion(gen_preds, torch.ones_like(gen_preds))
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}] Loss D: {disc_loss:.4f}, Loss G: {gen_loss:.4f}")
        save_image(fake_imgs.view(-1, 3, img_size, img_size), f"Code\Tree Age Predictor & Generator\generated_data\epoch_{epoch}.png", normalize=True)



def generate_images(age, num_images, save_dir=r"Code\Tree Age Predictor & Generator\generated_data"):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, noise_dim).to(device)
            age_tensor = torch.tensor([age], dtype=torch.float).to(device)
            fake_img = generator(noise, age_tensor).view(3, img_size, img_size)
            save_image(fake_img, os.path.join(save_dir, f"tree_age_{age}_{i}.png"), normalize=True)
