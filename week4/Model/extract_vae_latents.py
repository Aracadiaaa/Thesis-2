import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae_model import ConvVAE  # make sure this file has the full VAE class

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ========== DATASET ==========
data_dir = "Data\dataset_images"  # your parent folder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # no shuffle for consistent ordering

# ========== LOAD MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE().to(device)
model.load_state_dict(torch.load("vae_trained.pth", map_location=device, weights_only=True))
model.eval()  # set to evaluation mode

# ========== LATENT VECTOR EXTRACTION ==========
all_latents = []

with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)
        enc = model.encoder(images)
        enc = enc.view(enc.size(0), -1)
        mu = model.fc_mu(enc)
        logvar = model.fc_logvar(enc)
        z = model.reparameterize(mu, logvar)  # latent vector
        all_latents.append(z.cpu())

# Concatenate all latent vectors into a single tensor
all_latents = torch.cat(all_latents, dim=0)
print("Latent vectors shape:", all_latents.shape)

# Save the latent vectors
torch.save(all_latents, "vae_latents.pt")