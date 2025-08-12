import torch
import torch.nn as nn
import numpy as np
from openood.postprocessors import BasePostprocessor
from openood.postprocessors.autoencoder_postprocessor import Autoencoder  # Assuming you have an Autoencoder model defined

# You can use any normalizing flow implementation. Here is a simple RealNVP block.
class RealNVPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Linear(dim, dim)
        self.translate = nn.Linear(dim, dim)

    def forward(self, x):
        s = self.scale(x)
        t = self.translate(x)
        z = x * torch.exp(s) + t
        log_det_jacobian = s.sum(dim=1)
        return z, log_det_jacobian

    def inverse(self, z):
        s = self.scale(z)
        t = self.translate(z)
        x = (z - t) * torch.exp(-s)
        return x

class SimpleFlow(nn.Module):
    def __init__(self, dim, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([RealNVPBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        log_det = 0
        for block in self.blocks:
            x, ldj = block(x)
            log_det += ldj
        return x, log_det

    def log_prob(self, x):
        z, log_det = self.forward(x)
        # Assume standard normal prior
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * np.log(2 * np.pi)
        return log_pz + log_det

class FlowAutoencoderPostprocessor(BasePostprocessor):
    def __init__(self, args):
        super().__init__(args)
        # Load your trained autoencoder
        self.autoencoder = Autoencoder(latent_dim=32).cuda()
        self.autoencoder.load_state_dict(torch.load("/content/autoencoder_weights.pth"))
        self.autoencoder.requires_grad_(False)

        # Normalizing flow on latent space
        self.flow = SimpleFlow(dim=32, n_blocks=4).cuda()
        self.flow.load_state_dict(torch.load("/content/flow_weights.pth"))
        self.flow.eval()

    def inference(self, net, dataloader, progress=True):
        self.autoencoder.eval()
        self.flow.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                data = batch['data'].cuda()
                labels = batch['label']
                # Get latent representation
                encoded = self.autoencoder.encoder(data)
                # Flatten latent for flow
                latent_flat = encoded.view(encoded.size(0), -1)
                # Compute log-likelihood under flow
                log_prob = self.flow.log_prob(latent_flat)
                # OOD score: negative log-likelihood (lower = more OOD)
                scores = -log_prob
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        # Thresholding (as before)
        id_scores = all_scores[all_labels == 0]
        threshold = np.percentile(id_scores, 95) if len(id_scores) > 0 else np.percentile(all_scores, 95)
        pred = np.where(all_scores < threshold, 0, -1)

        return pred, all_scores, all_labels