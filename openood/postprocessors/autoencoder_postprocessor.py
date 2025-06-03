# filepath: /Users/i843890/Documents/Doutorado/my-fork/OpenMIBOOD/openood/postprocessors/autoencoder_postprocessor.py
from openood.postprocessors import BasePostprocessor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),  # 64x64x3 -> 64x64x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # 64x64x128 -> 32x32x128
            nn.GroupNorm(num_groups=32, num_channels=128),         # GroupNorm with 32 groups

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 32x32x128 -> 32x32x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # 32x32x64 -> 16x16x64
            nn.GroupNorm(num_groups=16, num_channels=64),          # GroupNorm with 16 groups

            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1), # 16x16x64 -> 16x16xlatent_dim
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)        # 16x16xlatent_dim -> 8x8xlatent_dim
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1), # 8x8xlatent_dim -> 8x8xlatent_dim
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),                           # 8x8xlatent_dim -> 16x16xlatent_dim
            nn.BatchNorm2d(latent_dim),

            nn.Conv2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),         # 16x16xlatent_dim -> 16x16x64
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),                           # 16x16x64 -> 32x32x64
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),                # 32x32x64 -> 32x32x128
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),                           # 32x32x128 -> 64x64x128

            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),                 # 64x64x128 -> 64x64x3
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderPostprocessor(BasePostprocessor):
    def __init__(self, args):
        super().__init__(args)
        # Initialize your autoencoder model here
        self.autoencoder = Autoencoder(latent_dim=32).cuda()
        # self.autoencoder.load_state_dict(torch.load("openood/postprocessors/autoencoder_weights.pth"))

        # Load the pre-trained weights for the autoencoder
        # UPDATE HERE with the correct path to your weights

        print("Loading autoencoder weights...")
        self.autoencoder.load_state_dict(torch.load("/content/autoencoder_weights.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_model_60_epochs_5e-4.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_hybrid_weights.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_mse_weights.pth"))
        
        self.autoencoder.requires_grad_(True)
        self.APS_mode = True
        self.hyperparam_search_done = True

    def inference(self, net, dataloader, progress=True):
        self.autoencoder.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                data = batch['data'].cuda()
                labels = batch['label']
                reconstructed = self.autoencoder(data)
                # Compute reconstruction error as OOD score
                scores = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3))
                # all_scores.append(scores.cpu())
                # all_scores.append(np.atleast_1d(scores.cpu().numpy()))

                # all_scores.append(scores.cpu().numpy().reshape(-1))
                all_scores.append(scores.cpu().detach().numpy())
                
                # all_labels.append(labels)
                # all_labels.append(labels.cpu().numpy().reshape(-1))
                all_labels.append(labels.cpu().detach().numpy())


        all_scores = np.concatenate(all_scores)
        # all_scores = torch.cat(all_scores)
        # all_labels = torch.cat(all_labels)
        all_labels = np.concatenate(all_labels)

        print("ID scores:", all_scores[all_labels == 0][:10])
        print("OOD scores:", all_scores[all_labels != 0][:10])

        # import matplotlib.pyplot as plt

        # plt.hist(all_scores[all_labels == 0], bins=50, alpha=0.5, label='ID')
        # plt.hist(all_scores[all_labels != 0], bins=50, alpha=0.5, label='OOD')
        # plt.legend()
        # plt.show()
        
        return np.zeros_like(all_labels), all_scores, all_labels