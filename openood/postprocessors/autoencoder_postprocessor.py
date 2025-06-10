# filepath: /Users/i843890/Documents/Doutorado/my-fork/OpenMIBOOD/openood/postprocessors/autoencoder_postprocessor.py
from openood.postprocessors import BasePostprocessor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class MultiLayerFeatureExtractor(nn.Module):
    def __init__(self, vgg_features_sequential, selected_layers):
        super().__init__()
        self.selected_layers = selected_layers
        # The 'vgg_features_sequential' argument is ALREADY the Sequential module containing the features.
        # We don't need to access '.features' again.
        self.vgg_layers = vgg_features_sequential
        self.layer_mapping = {
            'block1_conv1': 0,  'block1_conv2': 2,
            'block2_conv1': 5,  'block2_conv2': 7,
            'block3_conv1': 10, 'block3_conv2': 12, 'block3_conv3': 14, 'block3_conv4': 16,
            'block4_conv1': 19, 'block4_conv2': 21, 'block4_conv3': 23, 'block4_conv4': 25,
            'block5_conv1': 28, 'block5_conv2': 30, 'block5_conv3': 32, 'block5_conv4': 34,
        }

    def forward(self, x):
        features = []
        layer_indices_to_extract = [self.layer_mapping[l] for l in self.selected_layers]
        for name, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if name in layer_indices_to_extract:
                features.append(x)
        return features

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_features_sequential, selected_layers, selected_layer_weights):
        super().__init__()
        self.feature_extractor = MultiLayerFeatureExtractor(vgg_features_sequential, selected_layers)
        self.selected_layer_weights = selected_layer_weights

    def forward(self, input, recon):
        mean = torch.tensor([0.485, 0.456, 0.406], device=input.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input.device).view(1,3,1,1)
        input_norm = (input - mean) / std
        recon_norm = (recon - mean) / std

        feats_input = self.feature_extractor(input_norm)
        feats_recon = self.feature_extractor(recon_norm)

        # Compute per-sample perceptual loss
        losses = []
        for f1, f2, w in zip(feats_input, feats_recon, self.selected_layer_weights):
            # # Compute per-sample MSE (no reduction)
            # mse = F.mse_loss(f1, f2, reduction='none')
            # # Average over all but batch dimension
            # mse = mse.view(mse.size(0), -1).mean(dim=1)


            mse = F.mse_loss(f1, f2, reduction='none')
            mse = mse.view(mse.size(0), -1).sum(dim=1) / 1e6

            losses.append(w * mse)
        # Sum weighted losses for each sample
        total_loss = sum(losses)
        return total_loss  # shape: (batch_size,)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),  # 64x64x3 -> 64x64x128
            # nn.ReLU(),
            nn.ReLU(inplace=False),  # <-- Ensure inplace=False
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # 64x64x128 -> 32x32x128
            # nn.GroupNorm(num_groups=32, num_channels=128),         # GroupNorm with 32 groups
            nn.GroupNorm(num_groups=128, num_channels=128),  # or 1 group per channel (i.e., InstanceNorm)

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # 32x32x128 -> 32x32x64
            # nn.ReLU(),
            nn.ReLU(inplace=False),  # <-- Ensure inplace=False
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # 32x32x64 -> 16x16x64
            # nn.GroupNorm(num_groups=16, num_channels=64),          # GroupNorm with 16 groups
            nn.GroupNorm(num_groups=64, num_channels=64),  # or 1 group per channel (i.e., InstanceNorm)

            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1), # 16x16x64 -> 16x16xlatent_dim
            # nn.ReLU(),
            nn.ReLU(inplace=False),  # <-- Ensure inplace=False
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
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_weights.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_model_60_epochs_1e-3.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_model_10_epochs_1e-3_perceptual-diff.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_model_60_epochs_5e-4.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_hybrid_weights.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_mse_weights.pth"))
        # self.autoencoder.load_state_dict(torch.load("/content/autoencoder_model_60_epochs_1e-3_perceptual-samples.pth"))
        self.autoencoder.load_state_dict(torch.load("/content/autoencoder_model_60_epochs_1e-3_perceptual-samples-2.pth"))
        
        self.autoencoder.requires_grad_(False)


        # --- VGG19 Perceptual Loss Setup ---
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.cuda().eval()
        for param in vgg19.parameters():
            param.requires_grad = False

        # Choose layers and weights (example)
        # selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv4']
        # selected_layer_weights = [1.0, 0.75, 0.5]

        selected_layers = ['block2_conv2',"block3_conv3",'block4_conv3']
        selected_layer_weights = [4.0, 2.0, 1.0]
        # selected_layer_weights = [2.0 , 4.0 , 8.0]


        # Import your PerceptualLoss class here or define it above
        self.criterion = PerceptualLoss(vgg19, selected_layers, selected_layer_weights)

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
                # print("Reconstructed:", reconstructed)
                # print("Data:", data)

                # --- Use perceptual loss as OOD score ---
                scores = self.criterion(data, reconstructed)  # shape: (batch_size,)
                
                # For reporting average loss:
                # avg_score = scores.mean().item()
                # print("Average Perceptual Loss (Validation):", avg_score)
                # scores = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3))


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

        print("ID scores:", all_scores[all_labels == 0][:100])
        print("OOD scores:", all_scores[all_labels != 0][:100])

        id_scores = all_scores[all_labels == 0]
        
        # threshold = np.median(id_scores) if len(id_scores) > 0 else np.median(all_scores)
        threshold = id_scores.mean() + 1.0 * id_scores.std() if len(id_scores) > 0 else np.median(all_scores)
        
        # pred = 0 (ID) if score < threshold, -1 (OOD) otherwise
        pred = np.where(all_scores < threshold, 0, -1)

        # import matplotlib.pyplot as plt

        # plt.hist(all_scores[all_labels == 0], bins=50, alpha=0.5, label='ID')
        # plt.hist(all_scores[all_labels != 0], bins=50, alpha=0.5, label='OOD')
        # plt.legend()
        # plt.show()
        
        # return np.zeros_like(all_labels), all_scores, all_labels
        return pred, all_scores, all_labels
    

    # todo
# Try different combinations of VGG layers and weights.
# Consider combining perceptual loss with pixel-wise MSE.
# Investigate data normalization and augmentation.
# Explore other OOD scoring strategies.

# Tune the threshold (maybe use validation OOD set for optimal threshold).
# Try combining perceptual loss with pixel-wise MSE.
# Experiment with different VGG layers/weights or normalization strategies for further gains.


# Shallower layers: ['block1_conv2', 'block2_conv2', 'block3_conv3']
# Deeper layers: ['block3_conv4', 'block4_conv4', 'block5_conv4']
# Uniform weights: [1.0, 1.0, 1.0]
# Emphasize deeper: [1.0, 2.0, 4.0]
# Emphasize shallower: [4.0, 2.0, 1.0]