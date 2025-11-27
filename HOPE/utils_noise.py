import torch
import torch.nn.functional as F
import numpy as np

class MovingAverage:
    def __init__(self, num_data, num_classes, beta=0.99, warm_up=8):
        self.beta = beta
        self.num_data = num_data
        self.num_classes = num_classes
        # Initialize on CPU first, move to GPU when needed or in initial_bank
        self.label_bank = torch.ones((num_data, num_classes)) / num_classes
        self.warm_up = warm_up
        
    def initial_bank(self):
        # Reset to uniform distribution
        self.label_bank = torch.ones((self.num_data, self.num_classes)) / self.num_classes
        if torch.cuda.is_available():
            self.label_bank = self.label_bank.cuda()
    
    @torch.no_grad()
    def update(self, value, indices, epoch, weight=None):
        """
        Update the label bank and return the smoothed labels.
        """
        indices = indices.to(torch.long)
        
        # Ensure label_bank is on the correct device
        if self.label_bank.device != value.device:
            self.label_bank = self.label_bank.to(value.device)
            
        # Get current stored values
        y_pre = self.label_bank[indices, :]
        
        # Optional weighted update
        if weight is not None:
            value = (1-weight) * y_pre + weight * value
            
        # Moving Average Update
        # new_label = beta * old_label + (1-beta) * current_pred
        self.label_bank[indices, :] = self.beta * y_pre + (1 - self.beta) * value
        
        # Return the updated values
        return self.label_bank[indices, :]

def get_corrected_label(img_logits, pt_logits):
    """
    Correct pseudo labels using multi-modal consistency.
    Upgraded version: use confidence-weighted average when inconsistent.
    """
    soft_img = F.softmax(img_logits, dim=1)
    soft_pt = F.softmax(pt_logits, dim=1)
    
    # Joint prediction (simple average)
    joint = (soft_img + soft_pt) / 2
    
    # Check consistency
    img_hard = torch.argmax(soft_img, dim=1)
    pt_hard = torch.argmax(soft_pt, dim=1)
    agreement = (img_hard == pt_hard).float()[:, None] # (B, 1)
    
    # When inconsistent, use confidence-weighted average
    max_conf_img = torch.max(soft_img, dim=1, keepdim=True)[0]  # (B, 1)
    max_conf_pt = torch.max(soft_pt, dim=1, keepdim=True)[0]     # (B, 1)
    
    # Normalize weights (sum to 1)
    weight_img = max_conf_img / (max_conf_img + max_conf_pt + 1e-8)
    weight_pt = 1.0 - weight_img
    
    # Weighted average
    weighted = weight_img * soft_img + weight_pt * soft_pt
    
    # If consistent, trust joint prediction; if inconsistent, use weighted
    corrected = agreement * joint + (1 - agreement) * weighted
    
    return corrected
