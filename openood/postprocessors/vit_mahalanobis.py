import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
from openood.postprocessors import BasePostprocessor

class ViTFeatureExtractor(nn.Module):
    def __init__(self, ):
        super().__init__()
        # Load the full ViT model with weights and set to evaluation mode
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval().cuda()
        # Freeze all parameters
        for p in self.vit.parameters():
            p.requires_grad = False
        
        # Hook to extract features before the final classifier
        self.features = None
        def hook_fn(module, input, output):
            self.features = input[0]  # Get the input to the head (after encoder)
        
        self.vit.heads.register_forward_hook(hook_fn)

        # # Define a dictionary to store the output of the hooked layer
        # self.layer_outputs = {}

        # # Define the hook function
        # def hook_fn(module, input, output):
        #     self.layer_outputs['encoder_ln_output'] = output

        # # Register the hook on the final Layer Normalization layer after the encoder
        # self.vit.encoder.ln.register_forward_hook(hook_fn)

    # def forward(self, x):
    #     # x: (B, 3, H, W), should already be normalized to ImageNet stats
    #     with torch.no_grad():
    #         # Perform a standard forward pass through the ViT model
    #         # _ = self.vit(x)  # We don't need the final classification output

    #         # Retrieve the stored output from the hook
    #         # features = self.layer_outputs['encoder_ln_output']
    #         features = self.vit._process_input(x)
    #         features = self.vit.encoder(features)

    #         # Extract the CLS token feature (the first token)
    #         cls_token = features[:, 0, :]  # (B, hidden_dim)

    #         # Clear the stored output for the next forward pass
    #         # self.layer_outputs.clear()

    #     return cls_token
    def forward(self, x):
        with torch.no_grad():
            _ = self.vit(x)  # Run full forward pass
            features = self.features  # Extract features from hook
            cls_token = features[:, 0, :]  # Get CLS token
        return cls_token

class ViTMahalanobisPostprocessor(BasePostprocessor):
    def __init__(self, args):
        super().__init__(args)
        # Initialize ViT feature extractor
        self.feature_extractor = ViTFeatureExtractor()
        
        # Load pre-fitted Mahalanobis parameters
        # You should save these after calling fit_id_distribution() and load them here
        mahalanobis_data = torch.load("/content/vit_mahalanobis_params.pth",weights_only=False)
        self.mean = mahalanobis_data['mean']
        self.cov_inv = mahalanobis_data['cov_inv']
        
        self.APS_mode = True
        self.hyperparam_search_done = True

    def mahalanobis_distance(self, feats):
        # feats: numpy array (N, D)
        diff = feats - self.mean
        left = np.dot(diff, self.cov_inv)
        mdist = np.sqrt(np.sum(left * diff, axis=1))
        return mdist

    def inference(self, net, dataloader, progress=True):
        self.feature_extractor.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                data = batch['data'].cuda()
                labels = batch['label']
                
                # Extract ViT features
                feats = self.feature_extractor(data)
                
                # Compute Mahalanobis distance scores
                scores = self.mahalanobis_distance(feats.cpu().numpy())
                
                all_scores.append(scores)
                all_labels.append(labels.cpu().numpy())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        # Thresholding for predictions
        id_scores = all_scores[all_labels == 0]
        threshold = np.percentile(id_scores, 95) if len(id_scores) > 0 else np.percentile(all_scores, 95)
        pred = np.where(all_scores < threshold, 0, -1)

        return pred, all_scores, all_labels

# # Helper function to fit and save Mahalanobis parameters (run this once)
# def fit_and_save_mahalanobis_params(dataloader, save_path="/content/vit_mahalanobis_params.pth"):
#     """
#     Fit Mahalanobis parameters on ID data and save them for later use.
#     Run this function once on your training/ID data before using the postprocessor.
#     """
#     feature_extractor = ViTFeatureExtractor()
#     features = []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             data = batch['data'].cuda()
#             feats = feature_extractor(data)
#             features.append(feats.cpu())
    
#     features = torch.cat(features, dim=0).numpy()
#     mean = np.mean(features, axis=0)
#     cov = np.cov(features, rowvar=False) + 1e-6 * np.eye(features.shape[1])
#     cov_inv = np.linalg.inv(cov)
    
#     # Save parameters
#     torch.save({
#         'mean': mean,
#         'cov_inv': cov_inv
#     }, save_path)
    
#     print(f"Mahalanobis parameters saved to {save_path}")

# Usage example (run this once to prepare the parameters):
# fit_and_save_mahalanobis_params(your_id_dataloader)